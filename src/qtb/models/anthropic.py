"""Claude through the Anthropic Messages API.

Requires the optional ``anthropic`` dependency (``uv sync --extra anthropic``) and a key
in the environment. Never used in tests or CI: the tests pass a stand-in client.

Choices that matter for a benchmark:

- **No server-side fallback.** A refused request is recorded as a refusal. Letting the
  API rerun it on another model would mix two models' answers under one configuration.
- **Thinking is left on, at the configured effort.** On current Opus and Sonnet models it
  cannot be turned off; effort is the only control, so it is part of the configuration.
- **Prompt caching on every request.** The conversation grows by appending, so the
  prefix is stable and later steps read it from the cache.
- Assistant content goes back exactly as received, thinking blocks included.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qtb.models.base import Message, ModelTurn, ToolCall, Usage

MAX_TOKENS = 16_000


def _block_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    return block.to_dict()


class AnthropicModel:
    def __init__(self, model: str, effort: str = "high", client: Any = None) -> None:
        if client is None:
            import anthropic  # optional dependency, imported only for real runs

            client = anthropic.Anthropic()
        self.client = client
        self.model_id = model
        self.effort = effort
        self.name = f"{model}@{effort}"

    def respond(
        self, system: str, messages: Sequence[Message], tools: Sequence[dict[str, Any]]
    ) -> ModelTurn:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=list(messages),
            tools=[{**tool, "strict": True} for tool in tools],
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        content = [_block_dict(b) for b in response.content]
        text = "\n".join(b["text"] for b in content if b.get("type") == "text")
        calls = [
            ToolCall(b["id"], b["name"], dict(b.get("input") or {}))
            for b in content
            if b.get("type") == "tool_use"
        ]
        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return ModelTurn(content, text, calls, response.stop_reason or "", usage, response.model)
