import os
import uuid
from unittest.mock import MagicMock, patch

import litellm
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, CompletionUsage

from huggingface_hub import ChatCompletionOutput


def get_sync_example_files() -> list[str]:
    """
    Get list of synchronous example files to test
    """
    examples_dir = "examples/observers"
    if not os.path.exists(examples_dir):
        return []

    sync_files = []
    for f in os.listdir(examples_dir):
        if not f.endswith(".py"):
            continue

        filepath = os.path.join(examples_dir, f)
        with open(filepath) as file:
            content = file.read()
            if "async def" not in content and "await" not in content:
                sync_files.append(filepath)

    return sync_files


@pytest.fixture(scope="function")
def mock_clients():
    """Fixture providing mocked API clients"""

    def openai_fake_return(*args, **kwargs):
        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    message=ChatCompletionMessage(
                        content="", role="assistant", tool_calls=None, audio=None
                    ),
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                )
            ],
            model="gpt-4",
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=10, total_tokens=20
            ),
            created=1727238800,
            object="chat.completion",
            system_fingerprint=None,
        )

    def hf_fake_return(*args, **kwargs):
        return ChatCompletionOutput(
            id=str(uuid.uuid4()),
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            choices=[{"message": {"content": "Hello, world!"}}],
            created=1727238800,
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            system_fingerprint=None,
        )

    # Create base mock for other clients
    base_mock = MagicMock()
    base_mock.chat.completions.create = MagicMock(side_effect=openai_fake_return)

    hf_mock = MagicMock()
    hf_mock.chat.completions.create = MagicMock(side_effect=hf_fake_return)

    mocks = {
        # Sync clients
        "openai.OpenAI": patch("openai.OpenAI", return_value=base_mock),
        "litellm.completion": patch("litellm.completion", litellm.mock_completion),
        "aisuite.Client": patch("aisuite.Client", return_value=base_mock),
        "huggingface_hub.InferenceClient": patch(
            "huggingface_hub.InferenceClient", return_value=hf_mock
        ),
    }

    # Start all patches
    for mock in mocks.values():
        mock.start()

    yield

    # Stop all patches
    for mock in mocks.values():
        mock.stop()


@pytest.mark.parametrize("example_path", get_sync_example_files())
def test_sync_example_files(example_path, mock_clients):
    """Test that synchronous example files execute without errors"""
    if "async def" in open(example_path).read() or "await" in open(example_path).read():
        pytest.skip("Skipping async example in sync test")

    print(f"Executing sync example: {os.path.basename(example_path)}")

    try:
        with open(example_path) as f:
            exec(f.read())
    except Exception as e:
        pytest.fail(f"Failed to execute {os.path.basename(example_path)}: {str(e)}")
