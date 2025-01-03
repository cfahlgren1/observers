from argilla import TextQuestion  # noqa
from observers import wrap_openai
from observers.stores import ArgillaStore
from openai import OpenAI

api_url = "<argilla-api-url>"
api_key = "<argilla-api-key>"

store = ArgillaStore(
    api_url=api_url,
    api_key=api_key,
    # questions=[TextQuestion(name="text")], optional
)

openai_client = OpenAI()

client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)

print(response.choices[0].message.content)
