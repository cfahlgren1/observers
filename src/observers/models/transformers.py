import uuid
from typing import Any, Dict, List, Optional, Union

import transformers
from typing_extensions import Self

from observers.models.base import ChatCompletionObserver, ChatCompletionRecord
from observers.stores.datasets import DatasetsStore
from observers.stores.duckdb import DuckDBStore


class TransformersRecord(ChatCompletionRecord):
    client_name: str = "transformers"

    @classmethod
    def from_response(
        cls,
        response: Dict[str, Any] = None,
        error: Exception = None,
        **kwargs,
    ) -> Self:
        if not response:
            return cls(finish_reason="error", error=str(error), **kwargs)
        generated_text = response[0]["generated_text"][-1]
        return cls(
            id=str(uuid.uuid4()),
            assistant_message=generated_text.get("content"),
            tool_calls=generated_text.get("tool_calls"),
            raw_response=response,
            **kwargs,
        )


def wrap_transformers(
    client: transformers.TextGenerationPipeline,
    store: Optional[Union[DuckDBStore, DatasetsStore]] = None,
    tags: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> ChatCompletionObserver:
    return ChatCompletionObserver(
        client=client,
        create=client.__call__,
        format_input=lambda inputs, **kwargs: {"text_inputs": inputs, **kwargs},
        parse_response=TransformersRecord.from_response,
        store=store,
        tags=tags,
        properties=properties,
    )
