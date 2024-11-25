import os

from observers.observers import wrap_openai
from observers.stores import DatasetsStore
from openai import OpenAI

store = DatasetsStore(
    repo_name="qwen-2-5-traces",
    every=5,  # sync every 5 messages
    private=True,  # make the repo private
)

openai_client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/", api_key=os.environ["HF_TOKEN"]
)

client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)
