"""The agent loop: a model, one Python tool, a step limit, and a full transcript.

A *step* is one model response. After each response the harness runs every tool call
in it, in order, and sends all the results back in one user message. The run ends when
the model answers without calling a tool, refuses, runs out of output tokens, or uses
up its steps. The answer is whatever ``answer.json`` holds at that point; the harness
never writes it.

The model sees the request, the data dictionary and the file list, and nothing from
the key. Prompts never mention traps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qtb.models.base import Model, Usage
from qtb.task import ANSWER_FILE, DICTIONARY_FILE, Task
from qtb.tools import DEFAULT_TIMEOUT_S, PYTHON_TOOL, execute

SYSTEM_PROMPTS = {
    "plain": (
        "You are a quantitative researcher. You work in a directory that holds a research "
        "request, a data dictionary and data files. Use the python tool to inspect the "
        f"files and do the analysis. When you have the numbers, write them to `{ANSWER_FILE}` "
        "in the working directory, in the format the request gives, and then reply with a "
        "short written report of what you did. You cannot ask questions; if something is "
        "unclear, make the choice a careful researcher would make and say so in the report."
    ),
}
# A second harness configuration: the same model asked to review its work before
# answering. It names no specific problem.
SYSTEM_PROMPTS["review"] = SYSTEM_PROMPTS["plain"] + (
    " Before you write the answer, read the data dictionary closely and review your "
    "analysis for anything a careful senior researcher would object to."
)

# Paths an agent has no reason to touch. A run whose code mentions one is flagged; the
# harness does not block it (see tools.py on what is and is not isolated).
SUSPICIOUS = ("key.json", "/tasks/", "answer_key")


@dataclass(frozen=True)
class AgentConfig:
    model: Model
    prompt: str = "plain"
    max_steps: int = 30
    tool_timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def name(self) -> str:
        return f"{self.model.name}+{self.prompt}"


def opening_message(task: Task, workdir: Path) -> str:
    files = sorted(p.name for p in workdir.iterdir() if p.is_file())
    dictionary = (workdir / DICTIONARY_FILE).read_text()
    return (
        f"{task.request.rstrip()}\n\n---\n\n{dictionary.rstrip()}\n\n---\n\n"
        "Files in the working directory: " + ", ".join(f"`{f}`" for f in files) + "\n"
    )


class LLMAgent:
    """Runs one model configuration on one task in a given working directory."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.name = config.name

    def solve(self, workdir: Path, task: Task) -> dict[str, Any]:
        cfg = self.config
        system = SYSTEM_PROMPTS[cfg.prompt]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": opening_message(task, workdir)}
        ]
        events: list[dict[str, Any]] = []
        usage = Usage()
        stop = "step_limit"
        final_text = ""
        started = time.monotonic()
        for step in range(1, cfg.max_steps + 1):
            turn = cfg.model.respond(system, messages, [PYTHON_TOOL])
            usage = usage + turn.usage
            messages.append({"role": "assistant", "content": turn.content})
            event: dict[str, Any] = {
                "step": step,
                "stop_reason": turn.stop_reason,
                "text": turn.text,
                "usage": turn.usage.to_dict(),
                "tool_calls": [],
            }
            events.append(event)
            final_text = turn.text or final_text
            if turn.stop_reason in ("refusal", "max_tokens") and not turn.tool_calls:
                stop = turn.stop_reason
                break
            if not turn.tool_calls:
                stop = "end_turn"
                break
            results = []
            for call in turn.tool_calls:
                result = execute(call.name, call.input, workdir, cfg.tool_timeout_s)
                event["tool_calls"].append(
                    {
                        "id": call.id,
                        "name": call.name,
                        "input": call.input,
                        "output": result.output,
                        "is_error": result.is_error,
                        "exit_code": result.exit_code,
                        "timed_out": result.timed_out,
                    }
                )
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result.output,
                }
                if result.is_error:
                    block["is_error"] = True
                results.append(block)
            messages.append({"role": "user", "content": results})

        code = " ".join(str(c["input"].get("code", "")) for e in events for c in e["tool_calls"])
        return {
            "agent": self.name,
            "model": cfg.model.name,
            "prompt": cfg.prompt,
            "max_steps": cfg.max_steps,
            "steps": len(events),
            "tool_calls": sum(len(e["tool_calls"]) for e in events),
            "stop": stop,
            "usage": usage.to_dict(),
            "seconds": round(time.monotonic() - started, 2),
            "report": final_text,
            "suspicious": sorted({s for s in SUSPICIOUS if s in code}),
            "system": system,
            "events": events,
        }
