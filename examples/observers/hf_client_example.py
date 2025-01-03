import os

import observers
from huggingface_hub import InferenceClient


api_key = os.getenv("HF_TOKEN")

# Patch the HF client
hf_client = InferenceClient(token=api_key)
client = observers.wrap_hf_client(hf_client)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Write a function in Python that checks if a string is a palindrome.",
        }
    ],
)
