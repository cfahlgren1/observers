Stores are classes that sync these observations to different storage backends (like DuckDB or Hugging Face datasets).

### Supported Stores

| Store | Example | Annotate | Local | Free | UI filters | SQL filters |
|-------|---------|----------|-------|------|-------------|--------------|
| [Hugging Face Datasets](https://huggingface.co/docs/huggingface_hub/en/package_reference/io-management#datasets) | [example](https://github.com/cfahlgren1/observers/blob/main/examples/stores/datasets_example.py) | ❌ | ❌ | ✅ | ✅ | ✅ |
| [DuckDB](https://duckdb.org/) | [example](https://github.com/ParagEkbote/observers/blob/main/examples/stores/duckdb_example.py) | ❌ | ✅ | ✅ | ❌ | ✅ |
| [Argilla](https://argilla.io/) | [example](https://github.com/ParagEkbote/observers/blob/main/examples/stores/argilla_example.py) | ✅ | ❌ | ✅ | ✅ | ❌ |
| [OpenTelemetry](https://opentelemetry.io/) | [example](https://github.com/ParagEkbote/observers/blob/main/examples/stores/opentelemetry_example.py) | ︖* | ︖* | ︖* | ︖* | ︖* |
| [Honeycomb](https://honeycomb.io/) | [example](https://github.com/ParagEkbote/observers/blob/main/examples/stores/opentelemetry_example.py) | ✅ |❌| ✅ | ✅ | ✅ |

(*) These features for the OpenTelemetry store, depend upon the provider you use.


#### Hugging Face Datasets Store

To view and query Hugging Face Datasets, you can use the [Hugging Face Datasets Viewer](https://huggingface.co/docs/hub/en/datasets-viewer). You can [find example datasets here](https://huggingface.co/datasets?other=observers).

From within here, you can query the dataset using SQL or using your own UI. Take a look at this example:

```py
from observers.observers import wrap_openai
from observers.stores import DatasetsStore
from openai import OpenAI

store = DatasetsStore(
    repo_name="gpt-4o-traces",
    every=5,  # sync every 5 minutes
)

openai_client = OpenAI()

client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)

print(response.choices[0].message.content)

```

#### DuckDB Store

The default store is [DuckDB](https://duckdb.org/) and can be viewed and queried using the [DuckDB CLI](https://duckdb.org/#quickinstall). Take a look at this example:

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

#### Argilla Store

The Argilla Store allows you to sync your observations to [Argilla](https://argilla.io/). To use it, you first need to create a [free Argilla deployment on Hugging Face](https://docs.argilla.io/latest/getting_started/quickstart/). Take a look at this example: 

```py
from argilla import TextQuestion  # noqa
from observers.observers import wrap_openai
from observers.stores import ArgillaStore
from openai import OpenAI

api_url = "<argilla-api-url>"
api_key = "<argilla-api-key>"

store = ArgillaStore(
    api_url=api_url,
    api_key=api_key,
    # questions=[TextQuestion(name="text")], optional
)

openai_client = OpenAI()

client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)

print(response.choices[0].message.content)

```


#### OpenTelemetry Store

The OpenTelemetry "Store" allows you to sync your observations to any provider that supports OpenTelemetry!

Examples are provided for [Honeycomb](https://honeycomb.io), but any provider that supplies OpenTelemetry compatible environment variables should Just Work®, and your queries will be executed as usual in your provider, against _trace_ data coming from Observers.

To view the complete set of examples, please view the [example folder](https://github.com/cfahlgren1/observers/tree/main/examples/stores).