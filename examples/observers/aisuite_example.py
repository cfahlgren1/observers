import os

import aisuite as ai

from observers import wrap_aisuite


# Initialize AI Suite client
client = ai.Client()

# Wrap client to enable tracking
client = wrap_aisuite(client)

# Set API keys
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Define models to test
models = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"]

# Define conversation messages
messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

# Get completions from each model
for model in models:
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.75
    )
    print(response.choices[0].message.content)
