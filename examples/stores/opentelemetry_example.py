from observers.observers import wrap_openai
from observers.stores import OpenTelemetryStore
from openai import OpenAI


# Use your usual environment variables to configure OpenTelemetry
# Here's an example for Honeycomb
# export OTEL_SERVICE_NAME=<identifiable-service-name>
# export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
# export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
# export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=<ingest-key>"

store = OpenTelemetryStore()

openai_client = OpenAI()

client = wrap_openai(openai_client, store=store)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)
# The OpenTelemetryStore links multiple completions into a trace
response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me another joke."}],
)

# Now query your Opentelemetry Compatible observability store as you usually do!

print(response.choices[0].message.content)
