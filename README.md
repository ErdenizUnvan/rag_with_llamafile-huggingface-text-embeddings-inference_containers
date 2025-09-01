# rag_with_llamafile-huggingface-text-embeddings-inference_containers

@for docker host:

docker pull   eunvan/bge-tei:1.8

docker run -d --rm --name tei-bge \
  -p 8082:80 \
  eunvan/bge-tei:1.8

docker pull eunvan/qwen2.5-llamafile:latest

docker container run --name my-rag-llm -d --rm -p 8081:8080 eunvan/qwen2.5-llamafile:latest

@for windows pc: 

pip install -r requirements.txt

python gradio_doc_chat_llamafile.py

