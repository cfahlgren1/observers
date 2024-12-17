import asyncio
import os

from openai import AsyncOpenAI

from observers import wrap_openai


openai_client = AsyncOpenAI(
    base_url="https://api-inference.huggingface.co/v1/", api_key=os.getenv("HF_TOKEN")
)

client = wrap_openai(openai_client)


async def get_response() -> None:
    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": "Tell me a joke."}],
    )
    print(response)


asyncio.run(get_response())
