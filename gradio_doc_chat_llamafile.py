from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import gradio as gr
import warnings
import requests
import numpy as np
from typing import List, Union

warnings.filterwarnings('ignore')

# ------------------- Ayarlar -------------------
TEI_BASE_URL = "http://10.1.100.45:8082"           # TEI (BGE)
client = OpenAI(base_url="http://10.1.100.45:8081/v1", api_key="x")  # llamafile

# ------------------- TEI Embeddings (LC Embeddings + L2-norm) -------------------
class TEIEmbeddings(Embeddings):
    """LangChain Embeddings arayüzü + L2-normalize (cosine için).
       /embed bazı sürümlerde liste [[...]] döndürebildiğinden şema esnekliği sağlar."""
    def __init__(self, base_url: str, timeout: int = 60):
        self.embed_url = base_url.rstrip("/") + "/embed"          # TEI native
        self.embed_v1  = base_url.rstrip("/") + "/v1/embeddings"  # OpenAI uyumlu
        self.sess = requests.Session()
        self.timeout = timeout

    def _post(self, url: str, payload: dict):
        r = self.sess.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _normalize_schema(self, js: Union[list, dict]) -> List[List[float]]:
        # 1) Bazı TEI derlemeleri /embed için direkt liste döndürür: [[...], [...]]
        if isinstance(js, list):
            return js
        # 2) {"data":[{"embedding":[...]}]}  (hem /embed hem /v1/embeddings’te yaygın)
        if isinstance(js, dict) and "data" in js:
            return [row["embedding"] for row in js["data"]]
        # 3) {"embeddings":[[...],[...]]}
        if isinstance(js, dict) and "embeddings" in js:
            return js["embeddings"]
        raise ValueError(
            f"Unknown TEI response schema type={type(js)} "
            f"keys={list(js.keys()) if isinstance(js, dict) else 'n/a'}"
        )

    def _l2_normalize(self, vecs: List[List[float]]) -> List[List[float]]:
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).tolist()

    # ---- LangChain Embeddings API ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Önce /embed, olmazsa /v1/embeddings
        try:
            js = self._post(self.embed_url, {"inputs": texts, "truncate": True})
        except Exception:
            js = self._post(self.embed_v1, {"input": texts})
        vecs = self._normalize_schema(js)
        return self._l2_normalize(vecs)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ------------------- PDF Yükle & Split -------------------
documents = PyPDFLoader("nestle.pdf").load()
text_chunks = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documents)

# ------------------- Embedding & FAISS Index -------------------
# L2-normalize ettiğimiz için FAISS'in L2 mesafesi cosine ile uyumlu çalışır.
embeddings = TEIEmbeddings(base_url=TEI_BASE_URL)
vector_db = FAISS.from_documents(text_chunks, embeddings)

# ------------------- Soru-Cevap (RAG) -------------------
def answer_query(query: str) -> str:
    relevant_docs = vector_db.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = f"""Answer the question based on the context below.
If the question is unrelated to the context, then just return as "It is not related" and
do not say anything else!

Context:
{context}

Question: {query}
Answer:
"""
    try:
        resp = client.chat.completions.create(
            model="local",
            temperature=0.1,
            max_tokens=512,
            stop=["</s>"],
            messages=[{"role": "user", "content": prompt}]
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error from LLM: {e}"
    return answer

# ------------------- Gradio -------------------
my_interface = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Your Query"),
    outputs=gr.Textbox(label="Answer"),
    title="Nestlé HR Policy Chatbot (llamafile + TEI/BGE)",
    description="Ask questions about the PDF using TEI embeddings and a local llamafile LLM."
)

my_interface.launch(server_name="0.0.0.0", server_port=7860)
