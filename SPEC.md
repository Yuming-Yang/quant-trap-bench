# SPEC: QuantTrapBench

Status: accepted, 2026-11-26. The scope changed the same day: no paid model runs (see Scope change).

## Problem

An LLM agent asked to "evaluate this factor" will return a number. Whether the number is right depends on things the request does not mention: delisted stocks missing from the file, fundamentals dated by period end and not by publication, a factor chosen on the full sample. A human researcher is expected to notice these. Does an agent?

This project is a small benchmark that measures it. Each task gives an agent a dataset and a research request. Some datasets contain a planted trap. The agent is scored on whether its answer matches the value a careful researcher would report.

## Approach

### 1. Task format

A task is a directory with:

- a data bundle (Parquet or CSV files and a short data dictionary);
- a request in plain language, with one required numeric answer (for example, the annualized Sharpe ratio of a stated strategy, or the mean rank IC of a stated factor);
- a hidden answer key: the **corrected value**, the **naive value**, a tolerance, and the list of traps.

The corrected value is what the analysis gives when the trap is handled. The naive value is what it gives when the data is taken at face value. Both are computed by reference code when the task is generated, so both are known exactly.

The request never mentions traps. The data dictionary is truthful: it describes the columns as they are, including the detail that reveals the trap to a careful reader (for example, that a date column is the fiscal period end).

### 2. Trap families

| Family | Trap | What the data or setup looks like |
| --- | --- | --- |
| Data | Survivorship | Stocks that delisted are absent from the whole history |
| Data | Fundamentals not point-in-time | Values are dated by period end; a separate column holds the publication date |
| Data | Unadjusted corporate actions | A split appears as a −50% return; an adjustment factor column is provided |
| Data | Price lookahead | A feature column is built from the next day's close |
| Data | Zero costs | The request asks for net performance; a cost schedule is provided in a side file |
| Method | Full-sample selection | The request asks to "pick the best lookback and report its Sharpe" |
| Method | Best-of-many reporting | Many variants are supplied; the request asks for the performance of the best one |
| Method | Misaligned IC | Factor and return columns are on the same row but the return is contemporaneous |

Data traps reuse the scenario generator ideas from `backtest-bias-suite`. Method traps need a factor panel with a known true signal, which the synthetic panel in the factor research pipeline provides.

About a quarter of the tasks are **controls** with no trap. They measure false alarms.

### 3. Scoring

The primary score is deterministic and needs no judge model.

- **Trap task.** Full credit if the answer is within tolerance of the corrected value. No credit if it is within tolerance of the naive value. Anything else is recorded as "other" and counted separately, because a wrong answer for an unrelated reason is not evidence about the trap.
- **Control task.** Full credit if the answer is within tolerance of the true value. A miss here is a false alarm or a plain error.
- **Method traps.** Where the correct response is a caveat and not a different number (best-of-many reporting), the task asks for a second numeric answer that only a correct treatment yields, such as the deflated Sharpe ratio given the number of variants.

A secondary score records whether the agent's written report names the trap. It is judged by an LLM against the answer key, on a sample checked by hand, and is reported separately. It never changes the primary score.

### 4. Harness

Each run gives the agent a working directory with the task files, a Python execution tool and a step limit. The agent writes `answer.json`. Transcripts, tool calls and token counts are saved for every run.

Two scripted agents ship with the benchmark and run in CI at no cost:

- a **naive agent** that runs the textbook analysis on the data as given;
- an **oracle agent** that applies the reference correction.

They validate the benchmark itself: the naive agent must score zero on every trap task and full marks on every control, and the oracle must score full marks everywhere.

### 5. Reporting

For each model configuration: the share of trap tasks answered with the corrected value, the share answered with the naive value, the control-task accuracy, and each of these by trap family, with confidence intervals over repeated runs.

## Non-goals

- No real market data. Every task is synthetic with a known answer.
- Not a test of whether an agent can find profitable strategies.
- No leaderboard and no claim about which model is "best at quant research". The results describe behaviour on these tasks.
- No fine-tuning or prompt optimization against the benchmark.

## Acceptance criteria

- [x] 20 to 30 tasks, about a quarter of them controls, generated from seeds by one command.
- [x] Every task's corrected and naive values are computed by reference code, and the two differ by more than twice the tolerance.
- [x] The naive agent scores zero on all trap tasks and full marks on all controls; the oracle scores full marks on all tasks. Both run in CI.
- [x] The scorer has unit tests for each outcome: corrected, naive, other, malformed answer, missing answer.
- [ ] Results for at least three model or harness configurations, each with repeated runs. *Dropped by the author's decision (no paid runs).*
- [ ] The false-alarm rate on controls is reported next to the detection rate. *Reported for the two reference agents only; model rates dropped with the model runs.*
- [ ] The secondary, judged score is validated on a hand-labelled sample, and its agreement rate is reported. *Dropped: it needs model runs and a paid judge.*
- [x] Every number in the README comes from saved transcripts in the repository or from one documented command.
- [x] README: one-sentence problem, a worked example task, results with numbers, design trade-offs, Limitations. *Results are the reference agents' validation, not model results.*
- [x] `uv run pytest` and ruff pass in CI with no paid API calls.

## Scope change (2026-11-26, author's decision)

The author decided not to make any paid model calls. The project is finished as a benchmark that can be inspected and run, without model results. Milestone 5 (model runs) is dropped. Milestone 6 reports the reference agents' results, which cost nothing and validate the benchmark. The harness, the Claude adapter and the cost estimate stay in the repository, so a run can be made later by anyone who chooses to pay for it.

## Milestones

1. **Scaffold, task format, scorer.** With scorer tests.
2. **Data-trap tasks.** Generators, reference solutions, naive and oracle agents.
3. **Method-trap tasks and controls.** Depends on the synthetic panel from the factor research pipeline.
4. **Harness.** Agent loop, sandboxed working directory, transcript storage. Pilot on three tasks to measure tokens per run.
5. **Model runs.** Only after the author approves a cost estimate built from the pilot.
6. **Results and README.**

Each milestone is one branch and one PR.

## Dependencies (checked 2026-11-15)

- `backtest-bias-suite` (MIT) for scenario generation.
- The factor research pipeline's synthetic panel, for method traps. Milestones 1 and 2 do not need it.
- `inspect-ai` 0.3.290 (MIT) is a candidate harness. Its sandboxing normally relies on Docker, which is not installed on the development machine. See D2.
- Model APIs: not chosen. No prices were looked up for this draft.

## Cost

Milestones 1 to 4 cost nothing beyond a three-task pilot. Milestone 5 is the only paid step. Its estimate is tasks × configurations × repeats × measured tokens per run, and it is presented to the author before anything is launched.

## Related work

Found by search on 2026-11-15 and described from abstracts; none was run.

- **Look-Ahead-Bench** (arXiv 2601.13770) measures look-ahead bias that comes from a model's memory of market history. This project plants traps in the data and the method, on synthetic data that no model can have memorized.
- **BacktestBench** (arXiv 2605.17937) tests whether agents can automate backtesting on real exchange data. It measures task completion, not whether the agent notices a flawed setup.
- **The Alpha Illusion** (arXiv 2605.16895) is a position paper arguing that reported alpha from LLM trading agents is backtest evidence only.

A benchmark named ALPHAQT-BENCH was noted in earlier planning as closely related. It did not come up in this search and must be read before the README makes any claim of novelty.

## Limitations known in advance

- **Author bias.** Tasks, dictionaries and keys reflect one view of what a careful researcher does; results may favour models whose habits match it. Compare more than one model family. The README carries the same caveat.
- **Synthetic traps are cleaner than real ones.** Each task has at most two traps and a truthful data dictionary. Real data hides problems less politely.
- **Small task count.** With 20 to 30 tasks, per-family results rest on a handful of tasks each.
- **A numeric score cannot see reasoning.** An agent can reach the corrected value by luck or miss it through an unrelated slip. The "other" category and the transcripts are there to show how often.

## Open decisions

| ID | Decision | Proposal |
| --- | --- | --- |
| D1 | Names | Repository `quant-trap-bench`, import name `qtb` |
| D2 | Harness | A minimal harness written for this project, with a temporary working directory per run. Revisit `inspect-ai` if Docker is installed |
| D3 | Trap list | The eight above; the author adds or removes from research experience |
| D4 | Whether the data dictionary should reveal the trap | Yes. A trap that cannot be found from the materials tests guessing, not care |
| D5 | Models and budget | Chosen by the author after the pilot |
| D6 | Order relative to the factor research pipeline | Milestones 1 and 2 can start any time; milestone 3 waits for its synthetic panel |
| D7 | Agent-proposed factors (the optional layer from the pipeline's planning notes) | Out of scope for v1 |
