from observers import wrap_openai
from observers.stores import DatasetsStore
from openai import OpenAI

store = DatasetsStore(
    repo_name="gpt-4o-traces",
    every=5,  # sync every 5 minutes
)

openai_client = OpenAI()

client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)

print(response.choices[0].message.content)
