import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import huggingface_hub
from typing_extensions import Self
from dataclasses import asdict
from observers.models.base import (
    AsyncChatCompletionObserver,
    ChatCompletionObserver,
    ChatCompletionRecord,
)
from observers.stores.datasets import DatasetsStore

if TYPE_CHECKING:

    from observers.stores.duckdb import DuckDBStore


class HFRecord(ChatCompletionRecord):
    client_name: str = "hf_client"

    @classmethod
    def from_response(cls, response=None, error=None, **kwargs) -> Self:
        """Create a response record from an API response or error"""
        if not response:
            return cls(finish_reason="error", error=str(error), **kwargs)

        choices = response.get("choices", [{}])[0].get("message", {})
        usage = response.get("usage", {})

        return cls(
            id=response.id if response.id else str(uuid.uuid4()),
            model=response.get("model"),
            completion_tokens=usage.get("completion_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            total_tokens=usage.get("total_tokens"),
            assistant_message=choices.get("content"),
            finish_reason=response.get("choices", [{}])[0].get("finish_reason"),
            tool_calls=choices.get("tool_calls"),
            function_call=choices.get("function_call"),
            raw_response=asdict(response),
            **kwargs,
        )


def wrap_hf_client(
    client: huggingface_hub.InferenceClient | huggingface_hub.AsyncInferenceClient,
    store: Optional[Union["DuckDBStore", DatasetsStore]] = None,
    tags: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> ChatCompletionObserver:
    if isinstance(client, huggingface_hub.AsyncInferenceClient):
        return AsyncChatCompletionObserver(
            client=client,
            create=client.chat.completions.create,
            format_input=lambda inputs, **kwargs: {"messages": inputs, **kwargs},
            parse_response=HFRecord.from_response,
            store=store,
            tags=tags,
            properties=properties,
        )

    return ChatCompletionObserver(
        client=client,
        create=client.chat.completions.create,
        format_input=lambda inputs, **kwargs: {"messages": inputs, **kwargs},
        parse_response=HFRecord.from_response,
        store=store,
        tags=tags,
        properties=properties,
    )
