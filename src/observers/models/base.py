import datetime
import random
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
    arguments: Optional[Dict[str, Any]] = None

    messages: List[Message] = None
    assistant_message: Optional[str] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: str = None
    tool_calls: Optional[Any] = None
    function_call: Optional[Any] = None

    @classmethod
    def from_response(cls, response=None, error=None, model=None, **kwargs):
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
            "arguments",
        ]

    @property
    def duckdb_schema(self):
        return f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            model VARCHAR,
            timestamp TIMESTAMP,
            messages JSON,
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
            arguments JSON,
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
        return [
            "tool_calls",
            "function_call",
            "tags",
            "properties",
            "raw_response",
            "arguments",
        ]

    @property
    def image_fields(self):
        return []

    @property
    def text_fields(self):
        return []


class ChatCompletionObserver:
    """
    Observer that provides an interface for tracking chat completions.
    Args:
        client (`Any`):
            The client to use for the chat completions.
        create (`Callable[..., Any]`):
            The function to use to create the chat completions., eg `chat.completions.create` for OpenAI client.
        format_input (`Callable[[Dict[str, Any], Any], Any]`):
            The function to use to format the input messages.
        parse_response (`Callable[[Any], Dict[str, Any]]`):
            The function to use to parse the response.
        store (`Union["DuckDBStore", DatasetsStore]`, *optional*):
            The store to use to save the records.
        tags (`List[str]`, *optional*):
            The tags to associate with records.
        properties (`Dict[str, Any]`, *optional*):
            The properties to associate with records.
        logging_rate (`float`, *optional*):
            The logging rate to use for logging, defaults to 1
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
        logging_rate: Optional[float] = 1,
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
        self.logging_rate = logging_rate

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    def _log_record(
        self, response, error=None, model=None, messages=None, arguments=None
    ):
        record = self.parse_response(
            response,
            error=error,
            model=model,
            messages=messages,
            tags=self.tags,
            properties=self.properties,
            arguments=arguments,
        )
        if random.random() < self.logging_rate:
            self.store.add(record)
        return record

    def create(
        self,
        messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Creates a completion.

        Args:
            messages (`Dict[str, Any]`):
                The messages to send to the assistant.
            **kwargs:
                Additional arguments passed to the create function. If stream=True is passed,
                the function will return a generator yielding streamed responses.

        Returns:
            Any:
                The response from the assistant, or a generator if streaming.
        """
        response = None
        kwargs = self.handle_kwargs(kwargs)
        excluded_args = {"model", "messages", "tags", "properties"}
        arguments = {k: v for k, v in kwargs.items() if k not in excluded_args}
        model = kwargs.get("model")
        input_data = self.format_input(messages, **kwargs)

        if kwargs.get("stream", False):

            def stream_responses():
                response_buffer = []
                try:
                    for chunk in self.create_fn(**input_data):
                        yield chunk
                        response_buffer.append(chunk)
                    self._log_record(
                        response_buffer,
                        model=model,
                        messages=messages,
                        arguments=arguments,
                    )
                except Exception as e:
                    self._log_record(
                        response_buffer,
                        error=e,
                        model=model,
                        messages=messages,
                        arguments=arguments,
                    )
                    raise

            return stream_responses()

        try:
            response = self.create_fn(**input_data)
            self._log_record(
                response, model=model, messages=messages, arguments=arguments
            )
            return response
        except Exception as e:
            self._log_record(
                response, error=e, model=model, messages=messages, arguments=arguments
            )
            raise

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Handle and process keyword arguments for the API call.

        This method merges the provided kwargs with the default kwargs stored in the instance.
        It ensures that any kwargs passed to the method call take precedence over the default ones.
        """
        return {**self.kwargs, **kwargs}

    def __getattr__(self, attr: str) -> Any:
        if attr not in {"create", "chat", "messages"}:
            return getattr(self.client, attr)
        return getattr(self, attr)


class AsyncChatCompletionObserver(ChatCompletionObserver):
    """
    Async observer that provides an interface for tracking chat completions
    Args:
        client (`Any`):
            The async client to use for the chat completions.
        create (`Callable[..., Awaitable[Any]]`):
            The async function to use to create the chat completions.
        format_input (`Callable[[Dict[str, Any], Any], Any]`):
            The function to use to format the input messages.
        parse_response (`Callable[[Any], Dict[str, Any]]`):
            The function to use to parse the response.
        store (`Union["DuckDBStore", DatasetsStore]`, *optional*):
            The store to use to save the records.
        tags (`List[str]`, *optional*):
            The tags to include in the records.
        properties (`Dict[str, Any]`, *optional*):
            The properties to include in the records.
        logging_rate (`float`, *optional*):
            The logging rate to use for logging, defaults to 1
    """

    async def _log_record_async(
        self, response, error=None, model=None, messages=None, arguments=None
    ):
        record = self.parse_response(
            response,
            error=error,
            model=model,
            messages=messages,
            tags=self.tags,
            properties=self.properties,
            arguments=arguments,
        )
        if random.random() < self.logging_rate:
            await self.store.add_async(record)
        return record

    async def create(
        self,
        messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Create an async completion.

        Args:
            messages (`Dict[str, Any]`):
                The messages to send to the assistant.
        Returns:
            Any:
                The response from the assistant.
        """
        response = None
        kwargs = self.handle_kwargs(kwargs)
        excluded_args = {"model", "messages", "tags", "properties"}
        arguments = {k: v for k, v in kwargs.items() if k not in excluded_args}
        model = kwargs.get("model")
        input_data = self.format_input(messages, **kwargs)

        if kwargs.get("stream", False):

            async def stream_responses():
                response_buffer = []
                try:
                    async for chunk in await self.create_fn(**input_data):
                        yield chunk
                        response_buffer.append(chunk)
                    await self._log_record_async(
                        response_buffer,
                        model=model,
                        messages=messages,
                        arguments=arguments,
                    )
                except Exception as e:
                    await self._log_record_async(
                        response_buffer,
                        error=e,
                        model=model,
                        messages=messages,
                        arguments=arguments,
                    )
                    raise

            return stream_responses()

        try:
            response = await self.create_fn(**input_data)
            await self._log_record_async(
                response, model=model, messages=messages, arguments=arguments
            )
            return response
        except Exception as e:
            await self._log_record_async(
                response, error=e, model=model, messages=messages, arguments=arguments
            )
            raise

    async def __aenter__(self) -> "AsyncChatCompletionObserver":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.store.close_async()
