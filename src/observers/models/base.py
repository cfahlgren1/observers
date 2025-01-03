import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from typing_extensions import Self

from observers.base import Message, Record
from observers.stores.datasets import DatasetsStore


if TYPE_CHECKING:
    from argilla import Argilla

    from observers.stores.duckdb import DuckDBStore


@dataclass
class ChatCompletionRecord(Record):
    """
    Data class for storing chat completion records.
    """

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

    @classmethod
    def from_response(cls, response=None, error=None, **kwargs):
        """Create a response record from an API response or error"""
        pass

    @property
    def table_columns(self):
        return [
            "id",
            "model",
            "timestamp",
            "messages",
            "assistant_message",
            "completion_tokens",
            "prompt_tokens",
            "total_tokens",
            "finish_reason",
            "tool_calls",
            "function_call",
            "tags",
            "properties",
            "error",
            "raw_response",
            "synced_at",
        ]

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
                    description="The raw response from the API.",
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
        return f"{self.client_name}_records"

    @property
    def json_fields(self):
        return ["tool_calls", "function_call", "tags", "properties", "raw_response"]

    @property
    def image_fields(self):
        return []

    @property
    def text_fields(self):
        return []


class ChatCompletionObserver:
    """
    Observer that provides a clean interface for tracking chat completions
    Args:
        client (Any):
            The client to use for the chat completions.
        create (Callable[..., Any]):
            The function to use to create the chat completions., eg `chat.completions.create` for OpenAI client.
        format_input (Callable[[Dict[str, Any], Any], Any]):
            The function to use to format the input messages.
        parse_response (Callable[[Any], Dict[str, Any]]):
            The function to use to parse the response.
        store (Optional[Union["DuckDBStore", DatasetsStore]]):
            The store to use to save the records.
    """

    def __init__(
        self,
        client: Any,
        create: Callable[..., Any],
        format_input: Callable[[Dict[str, Any], Any], Any],
        parse_response: Callable[[Any], Dict[str, Any]],
        store: Optional[Union["DuckDBStore", DatasetsStore]] = None,
        tags: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.format_input = format_input
        self.parse_response = parse_response
        self.store = store or DatasetsStore.connect()
        self.tags = tags or []
        self.properties = properties or {}
        self.kwargs = kwargs

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    def create(
        self,
        messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Create a completion with optional custom formatters"""

        response = None
        try:
            kwargs = self.handle_kwargs(kwargs)

            input_data = self.format_input(messages, **kwargs)
            response = self.create_fn(**input_data)

            record = self.parse_response(
                response,
                tags=self.tags,
                properties=self.properties,
            )

            self.store.add(record)
            return response

        except Exception as e:
            record = self.parse_response(
                response,
                error=e,
                model=kwargs.get("model"),
                tags=self.tags,
                properties=self.properties,
            )
            self.store.add(record)
            raise

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Handle and process keyword arguments for the API call.

        This method merges the provided kwargs with the default kwargs stored in the instance.
        It ensures that any kwargs passed to the method call take precedence over the default ones.
        """
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs

    def __getattr__(self, attr: str) -> Any:
        if attr not in {"create", "chat", "messages"}:
            return getattr(self.client, attr)

        return getattr(self, attr)


class AsyncChatCompletionObserver(ChatCompletionObserver):
    """
    Async observer that provides a clean interface for tracking chat completions
    Args:
        client (Any):
            The async client to use for the chat completions.
        create (Callable[..., Awaitable[Any]]):
            The async function to use to create the chat completions.
        format_input (Callable[[Dict[str, Any], Any], Any]):
            The function to use to format the input messages.
        parse_response (Callable[[Any], Dict[str, Any]]):
            The function to use to parse the response.
        store (Optional[Union["DuckDBStore", DatasetsStore]]):
            The store to use to save the records.
    """

    def __init__(
        self,
        client: Any,
        create: Callable[..., Any],
        format_input: Callable[[Dict[str, Any], Any], Any],
        parse_response: Callable[[Any], Dict[str, Any]],
        store: Optional[Union["DuckDBStore", DatasetsStore]] = None,
        tags: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            client=client,
            create=create,
            format_input=format_input,
            parse_response=parse_response,
            store=store,
            tags=tags,
            properties=properties,
            **kwargs,
        )

    async def create(
        self,
        messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Create an async completion with optional custom formatters"""
        response = None
        try:
            kwargs = self.handle_kwargs(kwargs)

            input_data = self.format_input(messages, **kwargs)
            response = await self.create_fn(**input_data)

            record = self.parse_response(
                response,
                tags=self.tags,
                properties=self.properties,
            )

            await self.store.add_async(record)
            return response

        except Exception as e:
            record = self.parse_response(
                response,
                error=e,
                model=kwargs.get("model"),
                tags=self.tags,
                properties=self.properties,
            )
            await self.store.add(record)
            raise

    async def __aenter__(self) -> "AsyncChatCompletionObserver":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.store.close_async()
