from langchain_openai import ChatOpenAI

# Senin endpoint bilgilerin
llm = ChatOpenAI(
    model="local",
    api_key="x",   # dummy key, server kontrol etmiyorsa "x" yeterli
    base_url="http://10.1.100.45:8081/v1",
    temperature=0,
    max_tokens=512,
    stop=["</s>"]
)

# invoke çağrısı
response = llm.invoke("Summarize the differences between OSPF and IS-IS in 3 bullet points.")

print(response.content)
