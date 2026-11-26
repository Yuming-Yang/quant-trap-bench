# Decisions

Key choices, with the alternatives that were considered and dropped. Decisions D1 to D7 in the SPEC were accepted as proposed on 2026-11-26.

## Task format and scoring

**One task is a public folder plus a hidden key next to it.**
The harness copies only `public/` into the agent's working directory, so the key cannot be read by accident or by a curious agent. Dropped: one folder with the key in a dot-file, which an agent listing files with `ls -a` would find.

**Every requested number has a corrected value, a naive value and an absolute tolerance.**
A trap task is only useful if the trap moves the answer. The key refuses a trap task whose corrected and naive values are two tolerances apart or less, so no answer can be within tolerance of both, and a small numerical slip cannot carry a corrected answer onto the naive one.

**Outcome comes from one primary field.**
Some tasks ask for more than one number (the best-of-many task asks for the best Sharpe ratio and the probability that it reflects skill). Only the number that the trap changes decides the credit; the others are scored and kept for the transcript analysis. Dropped: all fields must be correct, which would mix the trap with unrelated slips in the supporting numbers.

**"Other" is a separate outcome from "naive".**
An answer that matches neither value is wrong for a reason the task did not set out to test, such as a different annualization. Counting it as naive would overstate how often agents fall for the trap; counting it as corrected would be wrong. Transcripts show which it was.

**Strict answer parsing.**
Values must be JSON numbers. Strings such as `"0.12"` or `"12%"` make the answer malformed. Lenient parsing would mean the scorer decides units on the agent's behalf. Requests state the unit of every number, so a strict parser does not penalize a careful agent.

**Tolerances are inclusive, with a relative slack of 1e-9.**
Without the slack, a value exactly one tolerance away can fall outside because of floating-point rounding.

## Data-trap tasks

**Generators written for this project, ideas taken from backtest-bias-suite.**
The traps in `backtest-bias-suite` (MIT) test a backtest engine on a few assets over a few weeks. Here the agent does the analysis itself, so the tasks need cross-sections large enough for a rank IC or an equal-weighted market return to be meaningful. The survivorship, split and lookahead tasks follow the same scenarios (T4, T5, T1 there) on new generators. Dropped: depending on that package, which would have tied the task data to its engine-oriented `Scenario` type.

**Keys come from the generators' arrays; the reference agents read the CSV files.**
The key and the oracle are two implementations of the same analysis on two representations of the data. The validity test requires them to agree to 1% of a tolerance, which catches a dictionary that does not say what the generator did, or a file that does not hold what the key assumed. Dropped: computing the key by running the oracle, which would make "the oracle scores full marks" true by construction.

**Requests define every statistic.**
Each request says how to annualize, which standard deviation to use, whether to subtract a risk-free rate and which dates to use. A careful answer should then differ from a naive one only because of the trap. This makes the tasks easier than real research requests, which is acceptable: the benchmark measures whether the trap is noticed, not whether conventions are guessed.

**Each trap is planted in a file, and the dictionary says so.**
SPEC decision D4. The survivorship file is a current-constituents snapshot with the delisted stocks in an archive file; the fundamentals carry a period end and a filing date; the raw closes come with a split factor; the signal's window is described; the cost schedule is in a side file. A request never names the problem. A test checks that no public text uses words such as "survivorship", "lookahead", "bias" or "trap".

**Seed fallback instead of tuning each draw.**
If the trap does not move the answer by more than two tolerances for a seed, generation moves to the next seed in a fixed sequence and records how many draws it took. With the current parameters every task is accepted on its first draw. Dropped: editing parameters per task until it passes, which hides the selection.

**Planted effects are strong enough to give clear answers.**
The lookahead trap's naive Sharpe ratio is around 50, the point-in-time naive IC around 0.14. A careful agent may notice that such a number is implausible, which is part of what is being measured. Weaker effects would make the corrected and naive values closer than two tolerances on many seeds.

**Controls are built to detect over-correction where one is natural.**
The split control lists the splits that are already applied, the lookahead control's signal is tradable as is, and the point-in-time control stores reports under the filing date. Tests show that applying the splits again, lagging the signal or adding a quarter's reporting lag moves the answer out of tolerance. The point-in-time control plants a larger IC (0.06) than the trap tasks so that a quarter's extra lag is detectable. The survivorship control has no natural over-correction; it measures only plain errors. The costs trap has no control.

## Method-trap tasks

**Built on factor-research-pipeline's synthetic panel, pinned to a commit.**
SPEC decision D6. The panel plants characteristics with a known IC and persistence and adds listings and delistings. The selection and IC tasks use its returns and its planted characteristic; the best-of-many task uses its deflated Sharpe ratio for the key. Both repositories are pinned by commit, so a change there cannot change a task silently.

**Full-sample selection is defined by a stated research protocol.**
"Pick the best lookback and report its Sharpe" has many defensible careful answers (walk-forward, cross-validation, a haircut). The data dictionary therefore states the team's protocol: choose on the research period, measure once on the holdout. The corrected value is what that protocol gives, so it is a single number. The holdout Sharpe ratio is a noisy draw and can be higher than the full-sample winner's; the task does not assume the bias has a sign on every seed.

**Best-of-many: the request names the probabilistic Sharpe ratio framework.**
Without it, Bonferroni, Sidak or a Sharpe haircut are all reasonable corrections and give different numbers, so a careful answer could not be told from an unrelated one. Naming the framework does not name the deflation. Answers that apply a different multiple-testing correction will land in "other"; transcripts show which ones they were. The variants are pure noise and independent, so the number of independent trials is the number of variants.

**Misaligned IC in both directions.**
A momentum signal that includes the current month inflates the row-by-row IC; a low-persistence planted signal deflates it. Testing only the inflated case would reward an agent that distrusts large ICs rather than one that checks the timing.

**Controls must detect their most likely over-correction.**
Where there is a natural over-correction (reporting only the last five years for the fixed-lookback control; shifting an already-forward return), the key records it as `false_alarm` and generation requires it to be more than two tolerances from the right answer. The selection control needed a second draw for this. The lookahead, split and point-in-time controls are covered by tests instead, because their over-corrections are far from the right answer on any seed.

**Catalog: 22 tasks, 6 controls (27%).**
Two trap tasks per trap, varied by seed, size and wording (the costs tasks differ in whether "net of costs" is said outright). Controls exist for six of the eight traps; costs and best-of-many have none, because a request without the trap would have to state the answer's basis outright.

## Harness

**A minimal harness of our own, a temporary directory per run.**
SPEC decision D2. `inspect-ai` normally sandboxes with Docker, which the development machine does not have. The harness is about 500 lines with comments: a model interface, one Python tool, a step limit, and a JSON record per run. The temporary directory isolates the files the agent should see; it is not a sandbox. Model-written code can read other files and use the network. Two mitigations: the tool's environment is stripped so API keys are not visible, and code that mentions the key file is flagged in the record. Revisit if Docker is installed or results are published.

**One tool, a fresh interpreter per call.**
A persistent REPL would be closer to a notebook but makes runs depend on hidden state and complicates timeouts. Files persist between calls; the tool description says so.

**The harness never writes the answer.**
If the model stops without writing `answer.json`, the run scores "missing". Parsing a number out of the final text would reward answers the model did not commit to.

**Step limit of 30 model responses.**
High enough for an analysis with a few false starts; low enough to bound the cost of a run that loops. Runs that hit it are recorded as such.

**No server-side fallback on refusals.**
The API can rerun a refused request on another model. For a benchmark that would mix models within one configuration, so refusals are recorded as refusals.

**Effort is part of a configuration.**
Thinking cannot be turned off on current Opus and Sonnet models, so effort is the control. Configurations are named `model@effort+prompt`.

**Two prompts: plain and review.**
The review prompt asks the model to check its analysis for anything a careful senior researcher would object to. It names no trap. Comparing it with the plain prompt on the same model measures how much a generic nudge helps.

**Messages use the Messages API block format internally.**
It is the only provider wired up. Another provider's adapter would translate to and from it.

**Runs are never overwritten.**
Each run is its own file under `runs/<configuration>/<task>/run-<k>.json`. Re-running needs a new output directory or a new repeat number, so a result cannot be replaced quietly.

**Paid runs need an explicit flag.**
`qtb run --agent claude` refuses to start without `--allow-paid`. The cost estimate is in `docs/cost-estimate.md`; nothing has been run.

## Scope

**No paid model runs (author's decision, 2026-11-26).**
The project ends as a benchmark with its validation, not as a study of models. The reference agents show that every task separates the careful answer from the naive one; what models do on it is left open. The adapter, `--allow-paid` guard and cost estimate remain, so the run is one command away for anyone who funds it. Dropped with it: model results, model false-alarm rates and the judged secondary score.
