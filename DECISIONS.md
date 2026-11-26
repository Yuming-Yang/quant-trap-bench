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
