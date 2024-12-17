import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import openai
from typing_extensions import Self

from observers.models.base import (
    AsyncChatCompletionObserver,
    ChatCompletionObserver,
    ChatCompletionRecord,
)
from observers.stores.datasets import DatasetsStore


if TYPE_CHECKING:
    from observers.stores.duckdb import DuckDBStore


class OpenAIRecord(ChatCompletionRecord):
    client_name: str = "openai"

    @classmethod
    def from_response(cls, response=None, error=None, **kwargs) -> Self:
        """Create a response record from an API response or error"""
        if not response:
            return cls(finish_reason="error", error=str(error), **kwargs)

        dump = response.model_dump()
        choices = dump.get("choices", [{}])[0].get("message", {})
        usage = dump.get("usage", {})
        model = kwargs.pop("model", dump.get("model"))
        return cls(
            id=response.id if response.id else str(uuid.uuid4()),
            model=model,
            completion_tokens=usage.get("completion_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            total_tokens=usage.get("total_tokens"),
            assistant_message=choices.get("content"),
            finish_reason=dump.get("choices", [{}])[0].get("finish_reason"),
            tool_calls=choices.get("tool_calls"),
            function_call=choices.get("function_call"),
            raw_response=dump,
            **kwargs,
        )


def wrap_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    store: Optional[Union["DuckDBStore", DatasetsStore]] = None,
    tags: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> ChatCompletionObserver:
    if isinstance(client, openai.AsyncOpenAI):
        return AsyncChatCompletionObserver(
            client=client,
            create=client.chat.completions.create,
            format_input=lambda inputs, **kwargs: {"messages": inputs, **kwargs},
            parse_response=OpenAIRecord.from_response,
            store=store,
            tags=tags,
            properties=properties,
        )

    return ChatCompletionObserver(
        client=client,
        create=client.chat.completions.create,
        format_input=lambda inputs, **kwargs: {"messages": inputs, **kwargs},
        parse_response=OpenAIRecord.from_response,
        store=store,
        tags=tags,
        properties=properties,
    )
