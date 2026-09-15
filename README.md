# GREENCLOSE

**Headline finding (walk-forward, 2018-01-01 to 2024-09-13, 1,654 out-of-sample days, 48 NSE names).**
There is a real but small cross-sectional signal, and it does not survive Indian transaction costs at
daily turnover. A linear model on 43 point-in-time features reaches a mean daily rank information
coefficient of **+0.039** with a Newey-West t-statistic of **+8.5**, comfortably past the t > 3 bar, and
it is almost entirely the short-term reversal effect that a one-line baseline already captures. Acting
on it earns **+11.6 basis points a day gross** on invested capital and pays **21.8 basis points a day**
in charges and slippage, for **-10.3 net**. The brief's break-even arithmetic said roughly 45 basis
points of gross daily alpha would be needed; the measurement came back at a quarter of that. The one
change that flips the sign is holding longer: at a ten-day hold, gross decays only to 8.2 basis points
while cost falls to 2.9, giving **+5.3 net** and a Sharpe near 0.7, which is still short of simply
holding the equal-weighted universe. The model predicts. The strategy does not pay. The leakage suite
passes 6/6 on the real data, so that conclusion is a measurement rather than a story.

*Numbers above are from the phases completed so far; the full five-seed ladder, the 1,000-simulation
random baseline and the single locked-holdout evaluation refine them. See `reports/` for the current
run and `research/research_log.jsonl` for the honest count of configurations tried.*

GREENCLOSE is a **paper-only research instrument** for one question: given only information available
at the close of day T, can a fixed 50-stock NSE universe be ranked by the probability that each stock
closes green on day T+1, and would acting on that ranking have paid after Indian transaction costs?

It contains **no broker integration, no order code, no credentials, and never will.**

The specification is [GREENCLOSE_BUILD_BRIEF.md](GREENCLOSE_BUILD_BRIEF.md) (v1.0). [PLAN.md](PLAN.md)
is the earlier research iteration kept for reference; its live-trading sections were deliberately removed
from scope.

## Setup

```
uv sync --all-groups
uv run pytest
```

## Pipeline

Each command caches its output, so any step can be re-run without redoing the ones before it.

```
uv run greenclose fetch          # yfinance -> immutable, content-hashed snapshot (or --from-csv DIR)
uv run greenclose validate       # reports/data_quality.html + a mechanical verdict line
uv run greenclose features       # point-in-time panel  (--check-truncation 10 to prove invariance)
uv run greenclose labels         # the six label variants
uv run greenclose leakage        # the six leakage checks on the REAL data; gates everything below
uv run greenclose baselines      # B0-B5 through the shared decision layer
uv run greenclose ladder         # model rungs 3-8 and the H2 experiment
uv run greenclose sweep          # --group decision | retrain | all
uv run greenclose report         # one self-contained HTML with the verdict block
uv run greenclose pin-model      # freeze one artefact for the forward runner
uv run greenclose forward        # tonight's predictions for every symbol; no orders
uv run greenclose holdout --confirm    # the locked holdout, once, ever
```

To run everything that remains in one go (about two hours, safe to interrupt and resume):

```
powershell -ExecutionPolicy Bypass -File scriptsun_full.ps1
```

## How to read a result

Two questions live in this project and they are kept apart everywhere.

**Question A, did the model predict?** Mean daily rank IC with a Newey-West t-statistic, Precision@K
against the correct daily base rate, lift over that base rate, and the long-short decile spread. The
significance bar is t > 3.0, not 2.0.

**Question B, would it have made money?** Net return against the equal-weighted universe after the
itemised Indian charge stack, with a Deflated Sharpe Ratio that reads its trial count from
`research/research_log.jsonl`. Every reported Sharpe carries one.

A system can pass A and fail B. That is the expected outcome and it is a valid result.

## Why you can believe the numbers

- **Look-ahead is impossible to write by accident.** `BarStore.get_bars` takes `cutoff` as a
  keyword-only argument with no default, so omitting it is a `TypeError`.
- **Truncation invariance is bitwise.** Features built on data cut at any date T are byte-identical,
  for every row dated at or before T, to features built on the whole dataset. Verified at ten random
  cut-offs on the real data.
- **The harness measures signal of the right size.** A synthetic feature injected at a known
  correlation of 0.10 comes back at 0.0965, so a scaling or alignment bug would show up as a wrong
  magnitude, not merely a missing signal.
- **Nothing is fitted outside its fold.** Winsorisation, imputation, scaling, hyperparameter choice
  and probability calibration all happen on training rows only, and a test recomputes the fitted
  statistics to prove it.
- **The trial count is honest.** Every out-of-sample configuration, including every hyperparameter
  cell and every sweep cell, is one line in `research/research_log.jsonl`.
- **The last 24 months are locked.** Every other command stops at the holdout boundary. `holdout`
  refuses to run twice and records the touch.

## Layout

```
config/      base, charges (dated, with source), universe, calendar, grid
src/greenclose/
  domain/    typed objects and Protocols
  ingest/    yfinance and CSV sources, content-hashed snapshots, the cutoff-enforcing store
  features/  registry and the pure point-in-time builder
  labels/    the six forward-return targets
  models/    splitters, heads, walk-forward training, calibration, artefact registry
  strategy/  decide.py: the one decision function the backtest and the forward runner both call
  backtest/  cost model, fill simulator, portfolio ledger, walk-forward engine
  evaluation/ metrics, significance, baselines, ladder, sweeps, leakage, report, forward, holdout
research/    research_log.jsonl, leakage_suite.json, ladder.json, holdout_touch.log  (evidence)
data/        snapshots, features, labels, predictions, runs, models  (gitignored)
reports/     data_quality.html and one HTML per run  (gitignored)
```

## What this project will never contain

No broker integration, no order placement, no credentials, no live runner. Also out of scope by
design: deep learning, reinforcement learning, news and sentiment pipelines, fundamentals, Kelly
sizing, leverage, and hyperparameter searches beyond the fixed grid in `config/grid.yaml`. A test
asserts the absence of broker code on every run.
