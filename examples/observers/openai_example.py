import os

from openai import OpenAI

from observers import wrap_openai


openai_client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/", api_key=os.getenv("HF_TOKEN")
)

client = wrap_openai(openai_client)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)
print(response)
