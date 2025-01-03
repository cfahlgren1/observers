# stdlib features
import asyncio
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

# Actual dependencies
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, Tracer, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

# Observers internal interfaces
from observers.base import Record
from observers.stores.base import Store


def flatten_dict(d, prefix=""):
    """Flatten a python dictionary, turning nested keys into dotted keys"""
    flat = {}
    for k, v in d.items():
        if v:
            if type(v) is dict:
                if prefix:
                    flat.extend(flatten_dict(v, f"{prefix}.{k}"))
            else:
                if prefix:
                    flat[(f"{prefix}.{k}")] = v
                else:
                    flat[k] = v


def get_version():
    try:
        return version("observers")
    except PackageNotFoundError:
        return "unknown"


@dataclass
class OpenTelemetryStore(Store):
    """
    OpenTelemetry Store
    """

    # These are here largely to ease future refactors/conform to
    # the style of the other stores. They have defaults set in the constructor,
    # but, set here as well.
    tracer: Optional[Tracer] = None
    root_span: Optional[Span] = None
    exporter: Optional[SpanExporter] = None
    namespace: str = "observers.dev/observers"

    def __post_init__(self):
        if not self.tracer:
            provider = TracerProvider(
                resource=Resource.create(
                    {
                        "instrument.name": self.namespace,
                        "instrument.version": get_version(),
                    }
                ),
            )
            if not self.exporter:
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            else:
                provider.add_span_processor(BatchSpanProcessor(self.exporter))
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(self.namespace)
        if not self.root_span:
            # if we initialize a span here, then all subsequent 'add's can be
            # added to a continuous trace
            with self.tracer.start_as_current_span(f"{self.namespace}.init") as span:
                span.set_attribute("connected", True)
                self.root_span = span

    def add(self, record: Record):
        """Add a new record to the store"""
        with trace.use_span(self.root_span):
            with self.tracer.start_as_current_span(f"{self.namespace}.add") as span:
                # Split out to be easily edited if the record api changes
                event_fields = [
                    "assistant_message",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_tokens",
                    "finish_reason",
                    "tool_calls",
                    "function_call",
                    "tags",
                    "properties",
                    "error",
                    "model",
                    "timestamp",
                    "id",
                ]
                for field in event_fields:
                    data = record.__getattribute__(field)
                    if data:
                        if type(data) is dict:
                            intermediate = flatten_dict(data, field)
                            for k, v in intermediate:
                                span.set_attribute(k, v)
                        else:
                            span.set_attribute(field, data)
                # Special case for `messages` as it is a list of dicts
                messages = [str(message) for message in record.messages]
                span.set_attribute("messages", messages)

    @classmethod
    def connect(cls, tracer=None, root_span=None, namespace=None, exporter=None):
        """Create an ObservabilityStore, optionally starting from a prior tracer or trace,
        assigning a custom namespace, or setting an alternate exporter"""
        return cls(tracer, root_span, namespace, exporter)

    def _init_table(self, record: "Record"):
        """Initialize the dataset (no op)"""
        # We don't usually do this in otel, a dataset is (typically)
        # initialized by writing to it, but, it's part of the Store interface.

    async def add_async(self, record: Record):
        """Add a new record to the store asynchronously"""
        await asyncio.to_thread(self.add, record)
