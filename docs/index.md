<div align="center">

<h1>🤗🔭 Observers 🔭🤗</h1>

<h3 align="center">A Lightweight Library for AI Observability</h3>

</div>

![Observers Logo](https://raw.githubusercontent.com/cfahlgren1/observers/main/assets/observers.png)

## Installation

First things first! You can install the SDK with pip as follows:

```bash
pip install observers
```

Or if you want to use other LLM providers through AISuite or Litellm, you can install the SDK with pip as follows:

```bash
pip install observers[aisuite] # or observers[litellm]
```

Whenever you want to observer document information, you can use our Docling integration.

```bash
pip install observers[docling]
```

For open telemetry, you can install the following:

```bash
pip install observers[opentelemetry]
```

## Usage

To get started, you can run this example code below. It sends requests to a HF serverless endpoint and log the interactions into a Hub dataset, using the default store `DatasetsStore`.

The dataset will be pushed to your personal workspace (http://hf.co/{your_username}). 

We differentiate between observers and stores.To know more, check out the next section.

```py
from observers.observers import wrap_openai
from observers.stores import DuckDBStore
from openai import OpenAI

store = DuckDBStore()

openai_client = OpenAI()
client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)
```

## Contributing

We ❤️ open-source contributions.See [CONTRIBUTING](https://github.com/cfahlgren1/observers/blob/main/CONTRIBUTING.md)
for more details.