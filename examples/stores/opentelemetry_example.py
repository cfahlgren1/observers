import os

from openai import OpenAI

from observers import wrap_openai
from observers.stores.opentelemetry import OpenTelemetryStore


# Use your usual environment variables to configure OpenTelemetry
# Here's an example for Honeycomb
os.environ.setdefault("OTEL_SERVICE_NAME", "llm-observer-example")
os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "https://api.honeycomb.io")

# Note: Keeping the sensitive ingest key in actual env vars, not in code
# export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=<api-key>"

store = OpenTelemetryStore()

openai_client = OpenAI()

client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Tell me a joke."}]
)

# The OpenTelemetryStore links multiple completions into a trace
response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Tell me another joke."}]
)
# Now query your Opentelemetry Compatible observability store as you usually do!
