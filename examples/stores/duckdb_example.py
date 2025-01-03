from observers import wrap_openai
from observers.stores import DuckDBStore
from openai import OpenAI

store = DuckDBStore()

openai_client = OpenAI()

client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)
