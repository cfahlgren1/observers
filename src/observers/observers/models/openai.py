import datetime
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from observers.observers.base import Message, Record
from observers.stores.duckdb import DuckDBStore
from openai import OpenAI

if TYPE_CHECKING:
    from argilla import Argilla
    from observers.stores.datasets import DatasetsStore


@dataclass
class OpenAIResponseRecord(Record):
    """
    Data class for storing OpenAI API response information
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    messages: List[Message] = None
    assistant_message: Optional[str] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: str = None
    tool_calls: Optional[Any] = None
    function_call: Optional[Any] = None
    tags: List[str] = None
    properties: Dict[str, Any] = None
    error: Optional[str] = None
    raw_response: Optional[Dict] = None

    @classmethod
    def create(cls, response=None, error=None, **kwargs):
        """Create a response record from an API response or error"""
        if not response:
            return cls(finish_reason="error", error=str(error), **kwargs)

        dump = response.model_dump()
        choices = dump.get("choices", [{}])[0].get("message", {})
        usage = dump.get("usage", {})

        return cls(
            id=response.id if response.id else str(uuid.uuid4()),
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

    @property
    def duckdb_schema(self):
        return f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            model VARCHAR,
            timestamp TIMESTAMP,
            messages STRUCT(role VARCHAR, content VARCHAR)[],
            assistant_message TEXT,
            completion_tokens INTEGER,
            prompt_tokens INTEGER,
            total_tokens INTEGER,
            finish_reason VARCHAR,
            tool_calls JSON,
            function_call JSON,
            tags VARCHAR[],
            properties JSON,
            error VARCHAR,
            raw_response JSON,
            synced_at TIMESTAMP
        )
        """

    def argilla_settings(self, client: "Argilla"):
        import argilla as rg
        from argilla import Settings

        return Settings(
            fields=[
                rg.ChatField(
                    name="messages",
                    description="The messages sent to the assistant.",
                    _client=client,
                ),
                rg.TextField(
                    name="assistant_message",
                    description="The response from the assistant.",
                    required=False,
                    client=client,
                ),
                rg.CustomField(
                    name="tool_calls",
                    template="{{ json record.fields.tool_calls }}",
                    description="The tool calls made by the assistant.",
                    required=False,
                    _client=client,
                ),
                rg.CustomField(
                    name="function_call",
                    template="{{ json record.fields.function_call }}",
                    description="The function call made by the assistant.",
                    required=False,
                    _client=client,
                ),
                rg.CustomField(
                    name="properties",
                    template="{{ json record.fields.properties }}",
                    description="The properties associated with the response.",
                    required=False,
                    _client=client,
                ),
                rg.CustomField(
                    name="raw_response",
                    template="{{ json record.fields.raw_response }}",
                    description="The raw response from the OpenAI API.",
                    required=False,
                    _client=client,
                ),
            ],
            questions=[
                rg.RatingQuestion(
                    name="rating",
                    description="How would you rate the response? 1 being the worst and 5 being the best.",
                    values=[1, 2, 3, 4, 5],
                ),
                rg.TextQuestion(
                    name="improved_response",
                    description="If you would like to improve the response, please provide a better response here.",
                    required=False,
                ),
                rg.TextQuestion(
                    name="context",
                    description="If you would like to provide more context for the response or rating, please provide it here.",
                    required=False,
                ),
            ],
            metadata=[
                rg.IntegerMetadataProperty(name="completion_tokens", client=client),
                rg.IntegerMetadataProperty(name="prompt_tokens", client=client),
                rg.IntegerMetadataProperty(name="total_tokens", client=client),
                rg.TermsMetadataProperty(name="model", client=client),
                rg.TermsMetadataProperty(name="finish_reason", client=client),
                rg.TermsMetadataProperty(name="tags", client=client),
            ],
        )

    @property
    def table_name(self):
        return "openai_records"

    @property
    def json_fields(self):
        return ["tool_calls", "function_call", "tags", "properties", "raw_response"]


def wrap_openai(
    client: OpenAI,
    store: Optional[Union[ "DatasetsStore", DuckDBStore]] = None,
    tags: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> OpenAI:
    """
    Wrap OpenAI client to track API calls in a Store.

    Args:
        client: OpenAI client instance
        store: Store instance for persistence. Creates new if None
        tags: Optional list of tags to associate with records
        properties: Optional dictionary of properties to associate with records
    """
    if store is None:
        store = DuckDBStore.connect()

    original_create = client.chat.completions.create

    def tracked_create(*args, **kwargs):
        try:
            response = original_create(*args, **kwargs)

            entry = OpenAIResponseRecord.create(
                response=response,
                messages=kwargs.get("messages"),
                model=kwargs.get("model"),
                tags=tags or [],
                properties=properties,
            )
            store.add(entry)
            return response

        except Exception as e:
            entry = OpenAIResponseRecord.create(
                error=e,
                messages=kwargs.get("messages"),
                model=kwargs.get("model"),
                tags=tags or [],
                properties=properties,
            )
            store.add(entry)
            raise

    client.chat.completions.create = tracked_create
    return client
