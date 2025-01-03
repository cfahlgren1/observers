import os

from litellm import completion

from observers import wrap_litellm


# Ensure you have both API keys set in environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Wrap the completion function to enable tracking
client = wrap_litellm(completion)

# Define models and messages
models = ["gpt-3.5-turbo", "claude-3-5-sonnet-20240620"]

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Get completions from each model
for model in models:
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.75
    )
    print(response.choices[0].message.content)
