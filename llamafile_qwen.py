from openai import OpenAI

client = OpenAI(base_url="http://10.1.100.45:8081/v1", api_key="x")

r = client.chat.completions.create(
    model="local",
    temperature=0,
    max_tokens=512,
    stop=["</s>"],
    messages=[
        {"role": "system", "content": "Answer shortly and technically. Respond in English."},
        {"role": "user", "content": "Summarize the differences between OSPF and IS-IS in 3 bullet points."}
    ]
)

print(r.choices[0].message.content)
