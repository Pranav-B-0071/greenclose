# GREENCLOSE

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Paper only](https://img.shields.io/badge/trading-paper%20only-critical.svg)
![Leakage suite 6/6](https://img.shields.io/badge/leakage%20suite-6%2F6-brightgreen.svg)

A paper-only research instrument that asks whether a fixed 50-stock NSE universe can be ranked by
next-day return using only information available at the close of day T, and whether acting on that
ranking would have paid after Indian transaction costs.

**It contains no broker integration, no order code, and no credentials. A test asserts their absence on
every run.**

## The finding

> There is a real cross-sectional signal. It is smaller than the fee.

Measured walk-forward on 1,654 out-of-sample days, 2018-01-01 to 2024-09-13, 48 NSE names, five seeds
per model.

| | Result | |
|---|---|:-:|
| Mean daily rank IC | **+0.0434**, Newey-West t = **+9.66**, 95% CI [+0.0347, +0.0526] | ✅ |
| Precision@3 lift over the daily base rate | **+3.18 pp**, 95% CI [+2.17, +4.52] | ✅ |
| Leakage suite on the real snapshot | **6 / 6 pass** | ✅ |
| Costs as a share of gross P&L | **26%** (1 of 5 seeds made no gross profit at all) | ✅ |
| Net return vs the equal-weighted universe | **−36.2 pp p.a.** (model +0.2%, universe +36.4%) | ❌ |
| Deflated Sharpe Ratio, N = 91 configurations tried | **0.000** | ❌ |

The model predicts. The strategy does not pay. That is a valid result, and it was the expected one.

Three things make the negative half of that convincing rather than disappointing:

**The signal is mostly one-day reversal.** `ret_1d` alone has an IC of −0.032 at t = −6.78. A
50-year-old idea captures three quarters of what the ensemble finds.

**Gradient boosting did not beat logistic regression.** Of seven paired bootstrap comparisons up the
ladder, only two cleared both the confidence interval and the spread across seeds. LightGBM, in a
head-to-head on identical folds, came in at IC +0.0391 against logistic regression's +0.0404.

**The cost gate does its job.** At the real Indian charge stack the abstention rule fires on
essentially every day: 120 trades in 1,654 days. Switch costs off as a diagnostic and the same signal
trades 4,061 times for +8.8% a year at a Sharpe of 1.59. The break-even arithmetic in the brief asked
for roughly 45 basis points of gross daily alpha; the measurement came back at about a quarter of that.

| Charges | Net p.a. | Sharpe | Trades | Abstention rate |
|---|---|---|---|---|
| × 0 (diagnostic only) | +8.8% | +1.59 | 4,061 | 53% |
| × 0.5 | +2.8% | +1.08 | 518 | 99% |
| **× 1 (real)** | **+0.8%** | **+0.46** | **120** | **100%** |
| × 1.5 | −0.3% | −0.23 | 38 | 100% |
| × 2 | −0.4% | −0.37 | 14 | 100% |

*One model (`lgbm_huber_calibrated`), one seed, the same cached predictions replayed through the
decision layer at each cost multiplier — so the signal is held fixed and only the fee moves.*

And one honesty note the report makes for itself: the per-period Sharpe across five seeds is
0.0157 ± 0.0142. The spread is as large as the effect, so the profitability result is not
distinguishable from seed noise. The ranking result, at an IC standard deviation of 0.0004 across
seeds, is.

### Two conventions worth knowing before you compare numbers

`net_return_pa` is total P&L over **initial** capital divided by years — a simple rate, not a compound
one. It is the right unit for comparing a strategy that deploys part of its account against one that
deploys all of it, and it is **not** comparable to a published CAGR. The equal-weighted universe's
+36.4% "p.a." is a total return of +241% over 6.6 years, which compounds at 20.4%. NIFTY buy-and-hold
over the same window is +14.2% compounded, which matches the index to the decimal and is one of the
cheapest sanity checks in the repository.

## Status

| Phase | State |
|---|---|
| P0–P4 scaffold, ingest, validation, features, labels, costs | complete |
| P5 walk-forward engine, sim broker, decision layer, baselines B1–B5 | complete |
| P6 leakage suite on synthetic **and real** data | complete, 6/6 |
| P7 model ladder, rungs 3–8, five seeds, H2 experiment | complete |
| P8 decision-layer and retrain sweeps | complete |
| P9 report and forward runner | complete |
| B0 random null distribution | **24 of 1,000 simulations** — the reported percentile is provisional |
| Forward log | 1 evening of 3 — an integrity check, never evidence |
| P10 locked holdout | **untouched**, and evaluable exactly once |

## Setup

```
uv sync --all-groups
uv run pytest
```

Ninety-one tests, offline, on a synthetic fixture. No network, no snapshot, no credentials.

## Pipeline

Every command caches its output, so any step can be re-run without redoing the ones before it.

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

Everything that remains, in one command. Budget four to five hours; it is safe to interrupt, because
the ladder caches its model fits:

```
powershell -ExecutionPolicy Bypass -File scripts/run_full.ps1
```

The retrain sweeps are three and a half of those hours: they are the only step that refits models, six
cells each a full monthly-retrain walk-forward, and nothing in them can reuse the ladder's cache. Pass
`-SkipSweeps` once they are done.

## How to read a result

Two questions live in this project and they are kept apart everywhere.

**Question A, did the model predict?** Mean daily rank IC with a Newey-West t-statistic, Precision@K
against the correct daily base rate, lift over that base rate, and the long-short decile spread. The
significance bar is t > 3.0, not 2.0.

**Question B, would it have made money?** Net return against the equal-weighted universe after the
itemised Indian charge stack, with a Deflated Sharpe Ratio that reads its trial count from
`research/research_log.jsonl`. Every reported Sharpe carries one.

A system can pass A and fail B. This one does.

## Why you can believe the numbers

- **Look-ahead is impossible to write by accident.** `BarStore.get_bars` takes `cutoff` as a
  keyword-only argument with no default, so omitting it is a `TypeError`, not a silent bug.
- **Truncation invariance is bitwise.** Features built on data cut at any date T are byte-identical,
  for every row dated at or before T, to features built on the whole dataset. Verified at ten random
  cut-offs on the real data. This is what licenses the engine to build the panel once and slice it.
- **The harness measures signal of the right size.** A synthetic feature injected at a known
  correlation of 0.10 comes back at 0.0965, so a scaling or alignment bug would show up as a wrong
  magnitude rather than as a missing signal.
- **Nothing is fitted outside its fold.** Winsorisation, imputation, scaling, hyperparameter choice
  and probability calibration all happen on training rows only, and a test recomputes the fitted
  statistics to prove it. The splitter has no `shuffle` argument to pass by mistake.
- **Replays are bit-identical.** The same configuration and seed reproduce the same replay parquet to
  the byte, across machines and across a refactor that made the engine 1.6× faster.
- **The trial count is honest.** Every out-of-sample configuration, including every hyperparameter cell
  and every sweep cell, is one line in `research/research_log.jsonl`, and the Deflated Sharpe Ratio
  reads N from that file. Re-running a configuration does not inflate it; trying a new one does.
- **The last 24 months are locked.** Every other command stops at the holdout boundary. `holdout`
  refuses to run twice and records the touch.

## Layout

```
config/            base, charges (dated, with source URL), universe, calendar, grid
src/greenclose/
  domain/          typed objects and Protocols
  ingest/          yfinance and CSV sources, content-hashed snapshots, the cutoff-enforcing store
  features/        registry and the pure point-in-time builder
  labels/          the six forward-return targets
  models/          splitters, heads, walk-forward training, calibration, artefact registry
  strategy/        decide.py: the one decision function the backtest and the forward runner share
  backtest/        cost model, fill simulator, portfolio ledger, walk-forward engine
  evaluation/      metrics, significance, baselines, ladder, sweeps, leakage, report, forward, holdout
research/          research_log.jsonl, leakage_suite.json, ladder.json, holdout_touch.log (evidence)
data/              snapshots, features, labels, predictions, runs, models  (gitignored)
reports/           data_quality.html and one HTML per run  (gitignored)
```

The specification is [GREENCLOSE_BUILD_BRIEF.md](GREENCLOSE_BUILD_BRIEF.md) (v1.0), and it is the
authority on every design decision here. [PLAN.md](PLAN.md) is the earlier research iteration, kept for
reference; its live-trading sections were deliberately removed from scope.

## What this project will never contain

No broker integration, no order placement, no credentials, no live runner. Also out of scope by design:
deep learning, reinforcement learning, news and sentiment pipelines, fundamentals, Kelly sizing,
leverage, and hyperparameter searches beyond the fixed grid in `config/grid.yaml`.

## Limitations, stated rather than buried

The universe is fixed and chosen today, so it carries survivorship bias; the report says so on every
run. Fills are simulated at the raw closing price as a proxy for the closing auction, which flatters a
strategy that trades at the close. Dividends are treated as reinvested through the adjusted series.
Two of the fifty symbols were dropped in validation and the reasons are recorded in
`config/universe.yaml`. With an average pairwise correlation near 0.3, the effective cross-sectional
sample on any given day is closer to 3 than to 48, which is why no t-statistic here is ever computed on
pooled rows.

## License

MIT. See [LICENSE](LICENSE).
