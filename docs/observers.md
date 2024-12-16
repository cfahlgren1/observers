Observers wrap generative AI APIs (like OpenAI or llama-index) and track their interactions. 

### Supported Observers

### OpenAI

[OpenAI](https://openai.com/) and every other LLM provider that implements the [OpenAI API message format](https://platform.openai.com/docs/api-reference).

The `wrap_openai` function allows you to wrap any OpenAI compliant LLM provider. Take a look at this example for more details.

```py
from observers.observers import wrap_openai
from openai import OpenAI

# Ollama is running locally at http://localhost:11434/v1
openai_client = OpenAI(base_url="http://localhost:11434/v1")

client = wrap_openai(openai_client)

response = client.chat.completions.create(
    model="llama3.1", messages=[{"role": "user", "content": "Tell me a joke."}]
)

```

### AISuite

[AISuite](https://github.com/andrewyng/aisuite), which is an LLM router by Andrew Ng and which maps to  a lot of [ LLM API providers](https://github.com/andrewyng/aisuite/tree/main/aisuite/providers) with a uniform interface.

To know more, take a look at this example:

```py
import os

from litellm import completion
from observers.observers import wrap_litellm

# Ensure you have both API keys set in environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Wrap the completion function to enable tracking
completion = wrap_litellm(completion)

# Define models and messages
models = ["gpt-3.5-turbo", "claude-3-5-sonnet-20240620"]

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Get completions from each model
for model in models:
    response = completion(model=model, messages=messages, temperature=0.75)
    print(response.choices[0].message.content)

```

### Litellm

[Litellm](https://docs.litellm.ai/docs/), which is a library that allows you to use a lot of different [LLM APIs](https://docs.litellm.ai/docs/providers) with a uniform interface.

To know more, take a look at this example:

```py
import os

from litellm import completion
from observers.observers import wrap_litellm

# Ensure you have both API keys set in environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Wrap the completion function to enable tracking
completion = wrap_litellm(completion)

# Define models and messages
models = ["gpt-3.5-turbo", "claude-3-5-sonnet-20240620"]

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Get completions from each model
for model in models:
    response = completion(model=model, messages=messages, temperature=0.75)
    print(response.choices[0].message.content)

```

### Docling

[Docling](https://github.com/docling/docling) parses documents and exports them to the desired format with ease and speed.

This observer allows you to wrap this and push popular document formats (PDF, DOCX, PPTX, XLSX, Images, HTML, AsciiDoc & Markdown) to the different stores.

To know more, take a look at this example:

```py
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from observers.observers.models.docling import wrap_docling

# Configure PDF pipeline options
pipeline_options = PdfPipelineOptions(
    images_scale=2.0,
    generate_page_images=True,
    generate_picture_images=True,
    generate_table_images=True,
)

# Set format options for PDF input
format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}

# Initialize and wrap document converter
converter = DocumentConverter(format_options=format_options)
converter = wrap_docling(converter, media_types=["pictures", "tables"])

# Convert single and multiple documents
url = "https://arxiv.org/pdf/2408.09869"
converted = converter.convert(url)
converted = converter.convert_all([url])

```

To view the complete set of examples, please view the [example folder](https://github.com/cfahlgren1/observers/tree/main/examples/observers).