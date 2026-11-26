# QuantTrapBench

Does an LLM agent notice the trap in a quant research task, or does it report the number the data gives at face value?

Status: work in progress. The task format, scorer and data-trap tasks are built; method traps, the agent harness and model runs are not. **No model has been run; there are no model results yet.**

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

Method traps (full-sample selection, best-of-many reporting, misaligned IC) are next.

## Reproduce

```bash
uv sync
uv run qtb generate --out tasks   # build every task from its seed
uv run qtb check --tasks tasks    # naive agent falls for every trap, oracle scores full marks
uv run pytest
```

## Design notes

See [DECISIONS.md](DECISIONS.md) and [SPEC.md](SPEC.md).
