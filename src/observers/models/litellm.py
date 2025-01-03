from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import litellm

from observers.models.base import AsyncChatCompletionObserver, ChatCompletionObserver
from observers.models.openai import OpenAIRecord

if TYPE_CHECKING:
    from observers.stores.argilla import ArgillaStore
    from observers.stores.datasets import DatasetsStore
    from observers.stores.duckdb import DuckDBStore


def wrap_litellm(
    client: Union[litellm.completion, litellm.acompletion],
    store: Optional[Union["DatasetsStore", "DuckDBStore", "ArgillaStore"]] = None,
    tags: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Wrap Litellm completion function to track API calls in a Store.

    Args:
        client: Litellm completion function
        store: Store instance for persistence. Creates new if None
        tags: Optional list of tags to associate with records
        properties: Optional dictionary of properties to associate with records
    """
    if client.__name__ == "acompletion":
        return AsyncChatCompletionObserver(
            client=client,
            create=client,
            format_input=lambda inputs, **kwargs: {"messages": inputs, **kwargs},
            parse_response=OpenAIRecord.from_response,
            store=store,
            tags=tags,
            properties=properties,
        )

    return ChatCompletionObserver(
        client=client,
        create=client,
        format_input=lambda inputs, **kwargs: {"messages": inputs, **kwargs},
        parse_response=OpenAIRecord.from_response,
        store=store,
        tags=tags,
        properties=properties,
    )
