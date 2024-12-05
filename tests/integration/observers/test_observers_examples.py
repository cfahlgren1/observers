import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice, CompletionUsage
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    InputAudio,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)


def get_example_files():
    """Get list of example files to test"""
    examples_dir = "examples/observers"
    if not os.path.exists(examples_dir):
        return []
    return [
        os.path.join(examples_dir, f)
        for f in os.listdir(examples_dir)
        if f.endswith(".py")
    ]


@pytest.fixture(scope="function")
def mock_clients():
    """Fixture providing mocked API clients"""

    def get_fake_return():
        return ChatCompletion(
            id=str(uuid.uuid4()),
            messages=[
                ChatCompletionSystemMessageParam(content=""),
                ChatCompletionUserMessageParam(
                    content=ChatCompletionContentPartTextParam(text="test")
                ),
                ChatCompletionAssistantMessageParam(content=""),
                ChatCompletionUserMessageParam(
                    content=ChatCompletionContentPartImageParam(image_url="image")
                ),
                ChatCompletionAssistantMessageParam(content=""),
                ChatCompletionUserMessageParam(
                    content=ChatCompletionContentPartInputAudioParam(
                        input_audio=InputAudio(data="audio", format="wav")
                    )
                ),
                ChatCompletionFunctionMessageParam(content=""),
                ChatCompletionToolMessageParam(content=""),
            ],
            choices=[
                Choice(
                    message=ChatCompletionMessage(
                        content="", role="assistant", tool_calls=None, audio=None
                    ),
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                ),
            ],
            model="gpt-4o",
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=10, total_tokens=20
            ),
            created=1727238800,
            object="chat.completion",
            system_fingerprint=None,
        )

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = (
        lambda *args, **kwargs: get_fake_return()
    )

    mock_aisuite = MagicMock()
    mock_aisuite.chat.completions.create.side_effect = (
        lambda *args, **kwargs: get_fake_return()
    )

    mock_litellm = MagicMock(side_effect=lambda *args, **kwargs: get_fake_return())

    mocks = {
        "openai.OpenAI": patch("openai.OpenAI", return_value=mock_openai),
        "aisuite.Client": patch("aisuite.Client", return_value=mock_aisuite),
        "litellm.completion": patch("litellm.completion", mock_litellm),
    }

    for mock in mocks.values():
        mock.start()

    yield

    for mock in mocks.values():
        mock.stop()


@pytest.mark.parametrize("example_path", get_example_files())
def test_example_files_execute(example_path, mock_clients):
    """Test that example files execute without errors"""
    if not get_example_files():
        pytest.skip("Examples directory not found")

    print(f"Executing {os.path.basename(example_path)}")

    try:
        exec(open(example_path).read())
    except Exception as e:
        pytest.fail(f"Failed to execute {os.path.basename(example_path)}: {str(e)}")
