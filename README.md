# QuantTrapBench

Does an LLM agent notice the trap in a quant research task, or does it report the number the data gives at face value?

Status: work in progress. The task format, scorer, all 22 tasks and the agent harness are built. **No model has been run; there are no model results yet.** The cost estimate for the runs is in [docs/cost-estimate.md](docs/cost-estimate.md) and awaits approval.

## What a task looks like

Each task gives an agent a few CSV files, a data dictionary and a research request with one required number. Some datasets contain a planted trap. The request never mentions it; the data dictionary describes the files truthfully, including the detail that reveals the trap to a careful reader.

Every task has a hidden key with two reference values, both computed when the task is generated:

- the **corrected** value, which a careful researcher would report;
- the **naive** value, which the analysis gives on the data taken at face value.

An answer within tolerance of the corrected value earns credit; one within tolerance of the naive value is recorded as having fallen for the trap; anything else is "other". About a quarter of the tasks are controls with no trap, which measure false alarms.

## Traps

| Family | Trap | How it is planted |
| --- | --- | --- |
| Data | Survivorship | The main returns file is a current-constituents snapshot; delisted stocks are in an archive file |
| Data | Fundamentals not point-in-time | Values carry a fiscal period end and a separate filing date |
| Data | Unadjusted splits | Raw closes with a cumulative split factor next to them |
| Data | Price lookahead | A momentum signal whose window ends at the next day's close |
| Data | Zero costs | The request asks for net performance; a cost schedule is in a side file |
| Method | Full-sample selection | "Pick the best lookback"; the dictionary states a research and holdout protocol |
| Method | Best-of-many reporting | Many skill-less variants; the request asks how likely the best one is to have skill |
| Method | Misaligned IC | Signal and return on the same row, but the return is contemporaneous |

22 tasks: two per trap, plus six controls without a trap (27%). The method tasks are built on the synthetic panel of [factor-research-pipeline](https://github.com/Yuming-Yang/factor-research-pipeline).

## Harness

Each run copies a task's public files into a fresh temporary directory and gives the model one tool, which runs Python there. The model works for up to 30 steps and must write `answer.json` itself. Every step's text, code, tool output and token counts are saved, one JSON file per run. Two scripted agents validate the benchmark at no cost: a naive agent that runs the textbook analysis on the data as given, and an oracle that applies the reference correction. Tests and CI drive the full loop with a scripted fake model.

## Results

Not run. Results will report, for each model configuration, the share of trap tasks answered with the corrected value and with the naive value, control accuracy, and each by trap family.

## Reproduce

```bash
uv sync
uv run qtb generate --out tasks   # build every task from its seed
uv run qtb check --tasks tasks    # naive agent falls for every trap, oracle scores full marks
uv run qtb run --agent oracle --tasks tasks --out runs
uv run qtb summarize runs/oracle
uv run pytest
```

Model runs need `uv sync --extra anthropic`, an API key and `--allow-paid`.

## Limitations

- **Synthetic traps are cleaner than real ones.** Each task has one trap, a truthful data dictionary, and a request that defines every statistic. Real data hides problems less politely.
- **Author bias.** The tasks were written with the help of a Claude model, and the proposed runs evaluate Claude models.
- **Small task count.** 16 trap tasks and 6 controls; per-family figures rest on a few tasks each.
- **A numeric score cannot see reasoning.** An agent can reach the corrected value by luck or miss it through an unrelated slip. The "other" category and the transcripts are there to show how often.
- **The harness is not a sandbox.** It isolates the task files, not the machine.

## Design notes

See [DECISIONS.md](DECISIONS.md) and [SPEC.md](SPEC.md).
