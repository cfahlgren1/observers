import os

from transformers import pipeline

import observers

model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    token=os.environ["HF_TOKEN"],
)
client = observers.wrap_transformers(pipe)
messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]
response = client.chat.completions.create(
    messages=messages,
    max_new_tokens=256,
)
print(response)
