"""The interface between the harness and a model, and a scripted fake for tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


@dataclass(frozen=True)
class ModelTurn:
    """One model response.

    ``content`` is the assistant message content exactly as it must be sent back on the
    next request (thinking blocks included, unchanged). ``text`` is the visible text.
    """

    content: list[dict[str, Any]]
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class Model(Protocol):
    name: str

    def respond(
        self, system: str, messages: Sequence[Message], tools: Sequence[dict[str, Any]]
    ) -> ModelTurn: ...


class FakeModel:
    """Replays a script. Each entry is a turn, or a function of the messages so far.

    Used in tests and CI so that the harness runs end to end without a paid API call.
    """

    def __init__(
        self,
        script: Sequence[ModelTurn | Callable[[Sequence[Message]], ModelTurn]],
        name: str = "fake",
    ) -> None:
        self.name = name
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    def respond(self, system, messages, tools) -> ModelTurn:
        self.requests.append({"system": system, "messages": list(messages), "tools": list(tools)})
        if not self._script:
            return say("(script exhausted)")
        entry = self._script.pop(0)
        return entry(messages) if callable(entry) else entry


def say(text: str, usage: Usage | None = None) -> ModelTurn:
    """A final turn with text and no tool call."""
    return ModelTurn([{"type": "text", "text": text}], text, [], "end_turn", usage or Usage(10, 5))


def run_code(
    code: str, call_id: str = "call_1", note: str = "", usage: Usage | None = None
) -> ModelTurn:
    """A turn that calls the python tool once."""
    content: list[dict[str, Any]] = []
    if note:
        content.append({"type": "text", "text": note})
    content.append({"type": "tool_use", "id": call_id, "name": "python", "input": {"code": code}})
    return ModelTurn(
        content,
        note,
        [ToolCall(call_id, "python", {"code": code})],
        "tool_use",
        usage or Usage(100, 20),
    )
