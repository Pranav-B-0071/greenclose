# GREENCLOSE

**Headline finding: not yet measured.** This paragraph will state the honest out-of-sample result
(rank IC with its Newey-West t-statistic and bootstrap CI, and net return versus the equal-weight
benchmark) once the walk-forward and the single locked-holdout evaluation have run. Until then there
is no claim of an edge.

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

```
uv run greenclose fetch          # yfinance -> content-hashed snapshot (or --from-csv DIR)
uv run greenclose validate       # reports/data_quality.html + verdict line
uv run greenclose features       # point-in-time feature panel
uv run greenclose labels
uv run greenclose baselines      # B0-B5 through the shared decision layer
uv run greenclose leakage        # the six leakage tests on the real snapshot
uv run greenclose ladder         # model rungs 0-8
uv run greenclose sweep
uv run greenclose report
uv run greenclose forward        # nightly prediction log; predictions only
uv run greenclose holdout --confirm   # once, ever
```
