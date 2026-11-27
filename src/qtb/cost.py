"""Token cost of runs, from a dated price table.

Prices are USD per million tokens, taken from https://platform.claude.com/docs/en/about-claude/pricing
on 2026-11-26 and re-checked on 2026-11-27 (standard API, global routing, 5-minute cache).
Check them again before approving spend; they change.

Before the pilot, tokens per run can only be assumed. ``scenario_usage`` turns an
assumed number of steps and tokens per step into the token counts an append-only,
prompt-cached agent loop produces. After the pilot, ``usage_cost`` prices the token
counts the transcripts actually recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_CHECKED = "2026-11-26"


@dataclass(frozen=True)
class Price:
    input: float
    cache_write: float  # 5-minute cache write
    cache_read: float
    output: float


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 5.00, 0.20, 20.00),
    "claude-sonnet-5-5": Price(2.00, 2.50, 0.10, 10.00),
    # Prompts up to 100,000 tokens; above that Haiku 5.5 costs five times as much.
    "claude-haiku-5-5": Price(0.10, 0.125, 0.01, 0.50),
}


def usage_cost(usage: dict[str, int], model: str) -> float:
    p = PRICES[model]
    return (
        usage.get("input_tokens", 0) * p.input
        + usage.get("cache_creation_input_tokens", 0) * p.cache_write
        + usage.get("cache_read_input_tokens", 0) * p.cache_read
        + usage.get("output_tokens", 0) * p.output
    ) / 1e6


@dataclass(frozen=True)
class Scenario:
    """Assumed shape of one run."""

    name: str
    steps: int
    opening_tokens: int  # system prompt, tool definitions, request and dictionary
    output_per_step: int  # thinking, text and tool code the model writes per step
    tool_result_per_step: int  # tool output sent back per step


def scenario_usage(s: Scenario) -> dict[str, int]:
    """Token counts for an append-only loop with automatic prompt caching.

    Request ``i`` (from 1) resends everything so far. The prefix written by earlier
    requests is read from the cache; what was added since (the last response and its
    tool result, or the opening message on the first request) is written to it.
    """
    growth = s.output_per_step + s.tool_result_per_step
    read = write = 0
    for i in range(1, s.steps + 1):
        if i == 1:
            write += s.opening_tokens
        else:
            read += s.opening_tokens + (i - 2) * growth
            write += growth
    return {
        "input_tokens": 0,
        "cache_creation_input_tokens": write,
        "cache_read_input_tokens": read,
        "output_tokens": s.steps * s.output_per_step,
    }
