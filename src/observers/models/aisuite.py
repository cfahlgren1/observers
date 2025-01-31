from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from observers.models.base import (
    AsyncChatCompletionObserver,
    ChatCompletionObserver,
)
from observers.models.openai import OpenAIRecord

if TYPE_CHECKING:
    from aisuite import Client

    from observers.stores.argilla import ArgillaStore
    from observers.stores.datasets import DatasetsStore
    from observers.stores.duckdb import DuckDBStore


class AisuiteRecord(OpenAIRecord):
    client_name: str = "aisuite"


def wrap_aisuite(
    client: "Client",
    store: Optional[Union["DatasetsStore", "DuckDBStore", "ArgillaStore"]] = None,
    tags: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
    logging_rate: Optional[float] = 1,
) -> Union[AsyncChatCompletionObserver, ChatCompletionObserver]:
    """Wraps Aisuite client to track API calls in a Store.

    Args:
        client (`Union[InferenceClient, AsyncInferenceClient]`):
            The HF Inference Client to wrap.
        store (`Union[DuckDBStore, DatasetsStore]`, *optional*):
            The store to use to save the records.
        tags (`List[str]`, *optional*):
            The tags to associate with records.
        properties (`Dict[str, Any]`, *optional*):
            The properties to associate with records.
        logging_rate (`float`, *optional*):
            The logging rate to use for logging, defaults to 1

    Returns:
        `ChatCompletionObserver`:
            The observer that wraps the Aisuite client.
    """
    return ChatCompletionObserver(
        client=client,
        create=client.chat.completions.create,
        format_input=lambda messages, **kwargs: kwargs | {"messages": messages},
        parse_response=AisuiteRecord.from_response,
        store=store,
        tags=tags,
        properties=properties,
        logging_rate=logging_rate,
    )
