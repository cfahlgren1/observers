import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from argilla import Argilla

from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepEndEvent,
    AgentChatWithStepStartEvent,
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatDeltaReceivedEvent,
    StreamChatErrorEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictStartEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryEndEvent,
    QueryStartEvent,
)
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index.core.instrumentation.events.span import (
    SpanDropEvent,
)
from llama_index.core.instrumentation.events.synthesis import (
    GetResponseStartEvent,
    SynthesizeEndEvent,
    SynthesizeStartEvent,
)
from observers.stores.base import Store
from observers.observers.base import Record

context_root: ContextVar[Union[Tuple[str, str], Tuple[None, None]]] = ContextVar(
    "context_root", default=(None, None)
)


class SimpleLlamaIndexHandler(BaseEventHandler):
    """
    Very simple handler for shipping llama_index spans and events with otel

    Attributes:
        store: Optional[Store] - The Observers store to use with LlamaIndex


    Usage:
        ```python
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.llms.openai import OpenAI

        from datasets_llama_index import DatasetsHandler

        store = OpenTelemetryStore()
        handler = SimpleLlamaIndexHandler(store=store)

        root_dispatcher = get_dispatcher()
        root_dispatcher.add_event_handler(handler)

        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY"))

        documents = SimpleDirectoryReader("../../data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=2)

        response = query_engine.query("What did the author do growing up?")
        ```
    """

    store: Optional[Store]

    def set_store(self, store: Store):
        self.store = store
        return self

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SimplerOtelHandler"

    def handle(self, event: BaseEvent) -> None:
        """
        Logic to handle different events.

        Args:
            event (BaseEvent): The event to be handled.

        Returns:
            None
        """
        record = LlamaIndexEventRecord.from_event(event)
        self.store.add(record, record.fields())


def clean(value: Any):
    if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
        try:
            return json.dumps(value)  # str
        except (TypeError, OverflowError):
            return str(value)
    if isinstance(value, dict):
        value = replace_empty_dicts(value)
        for inner_key, inner_value in value.items():
            value[inner_key] = clean(inner_value)
    if isinstance(value, list):
        for idx, inner_value in enumerate(value):
            value[idx] = clean(inner_value)
    try:
        json.dumps(value)
    except Exception:
        return str(value)
    return value


def replace_empty_dicts(self, data: Dict[str, Any]) -> Dict[str, Any]:
    def _replace_empty_dicts_recursive(value: Any) -> Any:
        if isinstance(value, dict):
            if not value:
                return None
            return {k: _replace_empty_dicts_recursive(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_replace_empty_dicts_recursive(item) for item in value]
        return value

    return _replace_empty_dicts_recursive(data)


@dataclass
class LlamaIndexEventRecord(Record):
    event_id: str = None
    event_type: str = None
    event_span_id: str = None
    event_timestamp: float = None
    event_tags: Optional[dict] = None
    task_id: Optional[str] = None
    step: Optional[str] = None
    input: Optional[str] = None
    step_output: Optional[str] = None
    user_msg: Optional[str] = None
    response: Optional[str] = None
    arguments: Optional[str] = None
    tool_name: Optional[str] = None
    tool_description: Optional[str] = None
    tool_openai_tool: Optional[dict] = None
    delta: Optional[str] = None
    exception: Optional[str] = None
    model_dict: Optional[dict] = None
    chunks: Optional[list[str]] = None
    embeddings: Optional[list[list[float]]] = None
    template_args: Optional[dict] = None
    output: Optional[str] = None
    template: Optional[str] = None
    output_cls: Optional[str] = None
    prompt: Optional[str] = None
    additional_kwargs: Optional[dict] = None
    messages: Optional[list[str]] = None
    str_or_query_bundle: Optional[str] = None
    nodes: Optional[list[str]] = None
    query: Optional[str] = None
    top_n: Optional[int] = None
    model_name: Optional[str] = None
    err_str: Optional[str] = None
    query_str: Optional[str] = None

    @classmethod
    def from_event(cls, event: BaseEvent):
        event_id = event.id_
        event_type = event.class_name()
        event_span_id = event.span_id
        event_timestamp = event.timestamp.timestamp()
        event_tags = event.tags if event.tags else None

        # just initialize all of them
        task_id = None
        step = None
        input = None
        step_output = None
        user_msg = None
        response = None
        arguments = None
        tool_name = None
        tool_description = None
        tool_openai_tool = None
        delta = None
        exception = None
        model_dict = None
        chunks = None
        embeddings = None
        template_args = None
        output = None
        template = None
        output_cls = None
        prompt = None
        additional_kwargs = None
        messages = None
        str_or_query_bundle = None
        nodes = None
        query = None
        top_n = None
        model_name = None
        err_str = None
        query_str = None

        if isinstance(event, AgentRunStepStartEvent):
            task_id = clean(event.task_id)  # str
            step = clean(event.step)  # opt str
            input = clean(event.input)  # opt str
        if isinstance(event, AgentRunStepEndEvent):
            step_output = clean(event.step_output)  # str
        if isinstance(event, AgentChatWithStepStartEvent):
            user_msg = clean(event.user_msg)  # str
        if isinstance(event, AgentChatWithStepEndEvent):
            response = clean(event.response)  # opt str
        if isinstance(event, AgentToolCallEvent):
            arguments = clean(event.arguments)  # str
            tool_name = clean(event.tool.name)  # str
            tool_description = clean(event.tool.description)  # str
            tool_openai_tool = clean(event.tool.to_openai_tool())  # dict
        if isinstance(event, StreamChatDeltaReceivedEvent):
            delta = clean(event.delta)  # str
        if isinstance(event, StreamChatErrorEvent):
            exception = clean(event.exception)  # str
        if isinstance(event, EmbeddingStartEvent):
            model_dict = clean(event.model_dict)  # dict
        if isinstance(event, EmbeddingEndEvent):
            chunks = clean(event.chunks)  # list(str)
            # embeddings = clean(event.embeddings)  # list(list(float))
            embeddings = None  # it's huge and doesn't make sense to send
        if isinstance(event, LLMPredictStartEvent):
            template = clean(event.template)  # str
            template_args = clean(event.template_arg)  # opt dict
        if isinstance(event, LLMPredictEndEvent):
            output = clean(event.output)  # str
        if isinstance(event, LLMStructuredPredictStartEvent):
            template = clean(event.template)
            template_args = clean(event.template_args)
            output_cls = clean(event.output_cls)  # str
        if isinstance(event, LLMStructuredPredictEndEvent):
            output = clean(event.output)
        if isinstance(event, LLMCompletionStartEvent):
            model_dict = clean(event.model_dict)
            prompt = clean(event.prompt)
            additional_kwargs = clean(event.additional_kwargs)
        if isinstance(event, LLMCompletionEndEvent):
            response = clean(event.response)
            prompt = clean(event.prompt)
        if isinstance(event, LLMChatInProgressEvent):
            messages = clean(event.messages)
            response = clean(event.response)
        if isinstance(event, LLMChatStartEvent):
            messages = clean(event.messages)
            additional_kwargs = clean(event.additional_kwargs)
            model_dict = clean(event.model_dict)
        if isinstance(event, LLMChatEndEvent):
            messages = clean(event.messages)
            response = clean(event.response)
        if isinstance(event, RetrievalStartEvent):
            str_or_query_bundle = clean(event.str_or_query_bundle)
        if isinstance(event, RetrievalEndEvent):
            str_or_query_bundle = clean(event.str_or_query_bundle)
            nodes = clean(event.node)
        if isinstance(event, ReRankStartEvent):
            query = clean(event.query)
            nodes = clean(event.nodes)
            top_n = clean(event.top_n)
            model_name = clean(event.model_name)
        if isinstance(event, ReRankEndEvent):
            nodes = clean(event.nodes)
        if isinstance(event, QueryStartEvent):
            query = clean(event.query)
        if isinstance(event, QueryEndEvent):
            response = clean(event.response)
            query = clean(event.query)
        if isinstance(event, SpanDropEvent):
            err_str = clean(event.err_str)
        if isinstance(event, SynthesizeStartEvent):
            query = clean(event.query)
        if isinstance(event, SynthesizeEndEvent):
            response = clean(event.response)
            query = clean(event.query)
        if isinstance(event, GetResponseStartEvent):
            query_str = clean(event.query_str)

        response = response.model_dump() if response else None
        str_or_query_bundle = (
            str_or_query_bundle.model_dump() if str_or_query_bundle else None
        )
        query = query.model_dump() if query else None
        template = template.model_dump() if template else None

        return cls(
            event_id,
            event_tags,
            None,
            err_str,
            None,
            event_id,
            event_type,
            event_span_id,
            event_timestamp,
            event_tags,
            task_id,
            step,
            input,
            step_output,
            user_msg,
            response,
            arguments,
            tool_name,
            tool_description,
            tool_openai_tool,
            delta,
            exception,
            model_dict,
            chunks,
            embeddings,
            template_args,
            output,
            template,
            output_cls,
            prompt,
            additional_kwargs,
            messages,
            str_or_query_bundle,
            nodes,
            query,
            top_n,
            model_name,
            err_str,
            query_str,
        )

    def to_dict(self) -> dict:
        {field: self.__getattribute__(field) for field in self.fields}

    def fields(self):
        return [
            "event_id",
            "event_type",
            "event_span_id",
            "event_timestamp",
            "event_tags",
            "task_id",
            "step",
            "input",
            "step_output",
            "user_msg",
            "response",
            "arguments",
            "tool_name",
            "tool_description",
            "tool_openai_tool",
            "delta",
            "exception",
            "model_dict",
            "chunks",
            "embeddings",
            "template_args",
            "output",
            "template",
            "output_cls",
            "prompt",
            "additional_kwargs",
            "messages",
            "str_or_query_bundle",
            "nodes",
            "query",
            "top_n",
            "model_name",
            "err_str",
            "query_str",
        ]

    def schema() -> dict:
        return {
            "event_id": "str",
            "event_type": "str",
            "event_span_id": "str",
            "event_timestamp": "float",
            "event_tags": "Optional(dict)",
            "task_id": "Optional(str)",
            "step": "Optional(str)",
            "input": "Optional(str)",
            "step_output": "Optional(str)",
            "user_msg": "Optional(str)",
            "response": "Optional(str)",
            "arguments": "Optional(str)",
            "tool_name": "Optional(str)",
            "tool_description": "Optional(str)",
            "tool_openai_tool": "Optional(dict)",
            "delta": "Optional(str)",
            "exception": "Optional(str)",
            "model_dict": "Optional(dict)",
            "chunks": "Optional(list(str))",
            "embeddings": "Optional(list(list(float)))",
            "template_args": "Optional(dict)",
            "output": "Optional(str)",
            "template": "Optional(str)",
            "output_cls": "Optional(str)",
            "prompt": "Optional(str),",
            "additional_kwargs": "Optional(dict)",
            "messages": "Optional(list(str))",
            "str_or_query_bundle": "Optional(str)",
            "nodes": "Optional(list(str))",
            "query": "Optional(str)",
            "top_n": "Optional(int)",
            "model_name": "Optional(str)",
            "err_str": "Optional(str)",
            "query_str": "Optional(str)",
        }

    def argilla_settings(self, client: "Argilla"):
        return []

    def duckdb_schema(self):
        return []

    def image_fields(self):
        return []

    def json_fields(self):
        return []

    def table_columns(self):
        return []

    def table_name(self):
        return []
