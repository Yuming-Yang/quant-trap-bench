# Cost estimate for model runs (milestone 5)

Written 2026-11-26. **Nothing has been run and nothing has been spent.** This estimate is for the author to approve, change or reject. The SPEC's plan was to measure tokens on a three-task pilot first; the pilot is itself a paid run, so it is part of what needs approval. The token figures below are therefore assumptions, not measurements.

## Prices

From Anthropic's pricing page (`https://platform.claude.com/docs/en/about-claude/pricing`), checked 2026-11-26. USD per million tokens, standard API, global routing, 5-minute prompt cache. Also in `src/qtb/cost.py`.

| Model | Input | Cache write | Cache read | Output |
| --- | --- | --- | --- | --- |
| Claude Opus 5.5 (`claude-opus-5-5`) | 4.00 | 5.00 | 0.20 | 20.00 |
| Claude Sonnet 5.5 (`claude-sonnet-5-5`) | 2.00 | 2.50 | 0.10 | 10.00 |
| Claude Haiku 5.5 (`claude-haiku-5-5`) | 0.10 | 0.125 | 0.01 | 0.50 |

Haiku 5.5's prices are five times higher for any request whose prompt exceeds 100,000 tokens. Only Claude models are listed because only a Claude adapter exists (see "Decisions for the author").

## Formula

Total = configurations × tasks × repeats × cost per run.

Cost per run comes from the harness's loop: every step resends the whole conversation, which grows by one response and one tool result. With automatic prompt caching, the earlier prefix is read from the cache and only the new part is written. `qtb.cost.scenario_usage` computes the token counts; `qtb.cost.usage_cost` prices them. The same `usage_cost` will price the pilot's recorded token counts.

## Assumed shape of one run

| Scenario | Steps | Opening message | Output per step (thinking, text, code) | Tool output per step | Largest prompt |
| --- | --- | --- | --- | --- | --- |
| Light | 8 | 2,500 | 800 | 500 | 10,300 |
| Central | 14 | 2,500 | 1,500 | 800 | 30,100 |
| Heavy | 30 (the step limit) | 2,500 | 3,000 | 1,500 | 128,500 |

## Proposed configurations

At least three are required by the SPEC. Proposed, for the author to change:

| ID | Model | Effort | Prompt |
| --- | --- | --- | --- |
| A | Claude Opus 5.5 | high | plain |
| B | Claude Opus 5.5 | high | review (asks the model to review its work; names no problem) |
| C | Claude Sonnet 5.5 | high | plain |
| D | Claude Haiku 5.5 | high | plain |

A against B isolates the prompt, A against C against D the model size.

## Estimate

Per run, USD:

| Scenario | A | B | C | D |
| --- | --- | --- | --- | --- |
| Light | 0.20 | 0.20 | 0.10 | 0.005 |
| Central | 0.62 | 0.62 | 0.31 | 0.017 |
| Heavy | 2.85 | 2.85 | 1.42 | 0.08 |

Totals, USD:

| Run | Size | Light | Central | Heavy |
| --- | --- | --- | --- | --- |
| Pilot | 3 tasks × 4 configurations × 1 repeat = 12 runs | 1.5 | 4.7 | 21.6 |
| Full | 22 tasks × 4 configurations × 3 repeats = 264 runs | 32.5 | 104.1 | 474.7 |

What could make it higher:

- **Caching not taking effect.** If every prompt were billed at the full input price, the full run would cost about 61, 238 and 1,677 USD in the three scenarios. The pilot will show the cache read share directly (`cache_read_input_tokens` in each transcript).
- **Haiku's long-prompt price** applies to the heavy scenario's late steps, so D's heavy figure is low by up to about 5×; D is small either way.
- **Retries** of failed requests by the SDK are not counted.

Not included: the secondary, judged score (an LLM reading each written report against the key). With 264 short reports it is a few dollars on Sonnet 5.5; it will be estimated separately before it runs.

## Proposed sequence

1. Approve the pilot: tasks `survivorship_1`, `point_in_time_control` and `selection_1` (one data trap, one control, one method trap), four configurations, one repeat. Expected about 5 USD, at most about 22 USD.
2. From the pilot's transcripts, replace the assumed scenario with measured tokens per run and re-price the full run with `qtb.cost.usage_cost`.
3. Approve the full run with the new figure.

Command, once approved (needs `uv sync --extra anthropic` and an API key in the environment):

```bash
uv run qtb run --agent claude --model claude-opus-5-5 --effort high --prompt plain \
  --task survivorship_1 --task point_in_time_control --task selection_1 \
  --out runs/pilot --allow-paid
```

## Decisions for the author (SPEC decision D5)

1. **Which models.** The SPEC's Limitations say results may favour the model family that helped write the tasks. These tasks were written with Claude's help, and the proposal evaluates only Claude models. Adding another family needs a new adapter (a small class with one method) and a price check. Whether to do that is the author's call.
2. **Budget ceiling** for the full run.
3. **Repeats.** Three per task gives a rough spread; with 16 trap tasks, per-family figures will still rest on a handful of tasks each.
4. **Effort.** "high" is proposed for every configuration. Lower effort is cheaper; whether it changes trap detection is itself a possible result.
