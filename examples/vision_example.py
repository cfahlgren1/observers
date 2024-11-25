from observers.observers import wrap_openai
from observers.stores import DatasetsStore
from openai import OpenAI

store = DatasetsStore(
    repo_name="gpt-4o-mini-vision-traces",
    every=5,  # sync every 5 messages
)

openai_client = OpenAI()
client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0].message.content)
