"""The Claude adapter, against a stand-in client. No network, no key."""

from types import SimpleNamespace

from qtb.models.anthropic import AnthropicModel
from qtb.tools import PYTHON_TOOL


class Block(dict):
    def to_dict(self):
        return dict(self)


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(content, stop_reason="tool_use"):
    usage = SimpleNamespace(
        input_tokens=1200,
        output_tokens=300,
        cache_read_input_tokens=900,
        cache_creation_input_tokens=None,
    )
    response = SimpleNamespace(
        content=[Block(b) for b in content],
        usage=usage,
        stop_reason=stop_reason,
        model="claude-opus-5-5",
    )
    return SimpleNamespace(messages=FakeMessages(response))


def test_request_shape():
    client = fake_client([{"type": "text", "text": "hi"}], "end_turn")
    model = AnthropicModel("claude-opus-5-5", effort="medium", client=client)
    model.respond("sys", [{"role": "user", "content": "go"}], [PYTHON_TOOL])
    kwargs = client.messages.calls[0]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == "sys"
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tools"][0]["strict"] is True
    assert kwargs["tools"][0]["input_schema"]["additionalProperties"] is False
    # No fallback to another model, no thinking override, no forced tool choice.
    assert not {"fallbacks", "betas", "thinking", "tool_choice"} & kwargs.keys()
    assert "strict" not in PYTHON_TOOL  # the shared definition is not mutated
    assert model.name == "claude-opus-5-5@medium"


def test_response_parsing_keeps_every_block():
    content = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_1", "name": "python", "input": {"code": "print(1)"}},
    ]
    turn = AnthropicModel("m", client=fake_client(content)).respond("s", [], [PYTHON_TOOL])
    assert turn.content == content
    assert turn.text == "Let me look."
    assert [(c.id, c.name, c.input) for c in turn.tool_calls] == [
        ("toolu_1", "python", {"code": "print(1)"})
    ]
    assert turn.stop_reason == "tool_use"
    assert turn.usage.input_tokens == 1200
    assert turn.usage.cache_read_input_tokens == 900
    assert turn.usage.cache_creation_input_tokens == 0
