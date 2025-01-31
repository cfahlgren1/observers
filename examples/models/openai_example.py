from openai import OpenAI

from observers import wrap_openai


openai_client = OpenAI()

client = wrap_openai(openai_client)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke in the voice of a pirate."}],
    stream=True,
    temperature=0.5,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
