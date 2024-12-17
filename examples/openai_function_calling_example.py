from openai import OpenAI

from observers import wrap_openai
from observers.stores import DatasetsStore

store = DatasetsStore(
    repo_name="gpt-4o-function-calling-traces",
    every=5,  # sync every 5 minutes
)

openai_client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    }
]

messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
    },
    {
        "role": "user",
        "content": "Hi, can you tell me the delivery date for my order? It's order 1234567890.",
    },
]


client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)
