from openai import OpenAI

from observers import wrap_openai


openai_client = OpenAI()

client = wrap_openai(openai_client)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke in the voice of a pirate."}],
    temperature=0.5,
)

print(response.choices[0].message.content)
