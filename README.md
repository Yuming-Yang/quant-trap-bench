# QuantTrapBench

Does an LLM agent notice the trap in a quant research task, or does it report the number the data gives at face value?

This repository is the benchmark: 22 tasks built from seeds, a deterministic scorer, an agent harness, and two scripted reference agents that validate every task. **No language model has been run on it**, so it reports no model results. The harness and a cost estimate for model runs ([docs/cost-estimate.md](docs/cost-estimate.md)) are included, so such a run is one command away.

## What a task looks like

Each task gives an agent a few CSV files, a data dictionary and a research request with one required number. Some datasets contain a planted trap. The request never mentions it; the data dictionary describes the files truthfully, including the detail that reveals the trap to a careful reader.

Every task has a hidden key with two reference values, both computed when the task is generated:

- the **corrected** value, which a careful researcher would report;
- the **naive** value, which the analysis gives on the data taken at face value.

An answer within tolerance of the corrected value earns credit; one within tolerance of the naive value is recorded as having fallen for the trap; anything else is "other". About a quarter of the tasks are controls with no trap, which measure false alarms.

## A worked example: `survivorship_1`

The request asks for the annualized mean return of the equal-weighted market portfolio, 2015 to 2024. The agent gets two files and a dictionary that describes them truthfully:

- `returns.csv`: "the stocks listed on the market on 2024-12-31, with their history back to January 2015", i.e. today's survivors only;
- `delisted_returns.csv`: the 85 stocks (of 200) that delisted during the period, with their delisting returns.

| Answer | `ew_mean` | Outcome |
| --- | --- | --- |
| Average `returns.csv` only, as the file name invites | 8.20% | naive: fell for survivorship |
| Combine both files, average whatever was listed each month | 5.18% | corrected |
| Tolerance | ±0.40 pp | |

The gap, 3.0 percentage points a year, is the survivorship bias in this sample: 7.5 tolerances wide, so no rounding or method detail can move a naive answer into the corrected band.

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

## Results: benchmark validation

These come from the two scripted reference agents and cost nothing. `uv run qtb check --tasks tasks` reproduces them, and after the runs in [Reproduce](#reproduce), `uv run qtb summarize runs/naive` and `uv run qtb summarize runs/oracle` print each agent's row; CI runs the check on every push.

| Agent | Trap tasks: corrected | Trap tasks: naive | Controls correct |
| --- | --- | --- | --- |
| Naive (textbook analysis on the data as given) | 0 of 16 | 16 of 16 | 6 of 6 |
| Oracle (reference correction) | 16 of 16 | 0 of 16 | 6 of 6 |

What this shows:

- **Every trap bites.** Taking the data at face value lands on the naive value in all 16 trap tasks, two per trap family.
- **Every trap is fixable from the materials.** The oracle reaches the corrected value everywhere, using only the files the agent sees.
- **Controls are not trick questions.** The naive analysis is correct on all 6 controls, so a model that "corrects" a control is raising a false alarm, not catching a subtle trap.
- **The two values are far apart.** The corrected and naive values are separated by at least 2.8 tolerances (survivorship_2) and up to 496 (lookahead_2), so the scorer never has to guess which one an answer was aiming at.

**Model results: not run**, by the author's decision. For a model, the benchmark would report the share of trap tasks answered with the corrected value and with the naive value, the false-alarm rate on controls, and each by trap family, with intervals over repeated runs. A run needs an API key and the explicit `--allow-paid` flag (see Reproduce).

## Reproduce

```bash
uv sync
uv run qtb generate --out tasks   # build every task from its seed
uv run qtb check --tasks tasks    # naive agent falls for every trap, oracle scores full marks
uv run qtb run --agent naive --tasks tasks --out runs
uv run qtb run --agent oracle --tasks tasks --out runs
uv run qtb summarize runs/naive
uv run qtb summarize runs/oracle
uv run pytest
```

A model run (not made here, and paid) needs `uv sync --extra anthropic`, `ANTHROPIC_API_KEY` in the environment, and `--allow-paid`; without the flag the command refuses to start. The estimate in [docs/cost-estimate.md](docs/cost-estimate.md) puts a full run (22 tasks, 4 configurations, 3 repeats) at roughly 30 to 480 USD depending on how long agents work.

## Limitations

- **Synthetic traps are cleaner than real ones.** Each task has one trap, a truthful data dictionary, and a request that defines every statistic. Real data hides problems less politely.
- **No model results.** The benchmark is validated, but how any model behaves on it is untested here.
- **Author bias.** Tasks, dictionaries and keys reflect one view of what a careful researcher does; results may favour models whose habits match it. Compare more than one model family.
- **Small task count.** 16 trap tasks and 6 controls; per-family figures rest on a few tasks each.
- **A numeric score cannot see reasoning.** An agent can reach the corrected value by luck or miss it through an unrelated slip. The "other" category and the transcripts are there to show how often.
- **The harness is not a sandbox.** It isolates the task files, not the machine.

## Design choices

- **A deterministic primary score.** An answer is a number compared with two reference values, so scoring needs no judge model and cannot drift. The cost is that the score cannot see reasoning; see Limitations.
- **Truthful data dictionaries.** Every trap can be found by a careful reader of the materials. A trap that could only be guessed would test luck, not care.
- **Controls mirror traps.** Each control uses the same data shape as a trap family without the flaw, so detecting a trap and over-correcting are measured on comparable tasks.
- **Synthetic data with known answers.** No model can have memorized the data, and both reference values are exact. Method traps reuse the synthetic factor panel of factor-research-pipeline, whose true signal is known.
- **A minimal harness instead of a framework.** One Python tool in a temporary directory and a step limit, because the development machine has no Docker. It isolates the task files, not the machine.

More detail, including what was dropped, is in [DECISIONS.md](DECISIONS.md).

## License

MIT. See [LICENSE](LICENSE).
