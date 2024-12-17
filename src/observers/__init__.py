import importlib.util
from typing import List

from .models.aisuite import wrap_aisuite
from .models.base import ChatCompletionObserver, ChatCompletionRecord
from .models.docling import wrap_docling
from .models.hf_client import wrap_hf_client
from .models.litellm import wrap_litellm

from .models.openai import OpenAIRecord, wrap_openai
from .models.transformers import TransformersRecord, wrap_transformers

from .stores.base import Store
from .stores.datasets import DatasetsStore

__all__: List[str] = [
    "ChatCompletionObserver",
    "ChatCompletionRecord",
    "TransformersRecord",
    "OpenAIRecord",
    "wrap_openai",
    "wrap_transformers",
    "DatasetsStore",
    "Store",
    "wrap_aisuite",
    "wrap_litellm",
    "wrap_docling",
    "wrap_hf_client",
]

if importlib.util.find_spec("argilla") is not None:
    from .stores.argilla import ArgillaStore

    __all__ += ["ArgillaStore"]

if importlib.util.find_spec("duckdb") is not None:
    from .stores.duckdb import DuckDBStore

    __all__ += ["DuckDBStore"]

OPTIONAL_DEPENDENCIES = {
    "argilla": ("argilla", "ArgillaStore"),
    "duckdb": ("duckdb", "DuckDBStore"),
}
