# GREENCLOSE — Build Brief

**A handoff specification for an implementing agent.**

> **Project:** GREENCLOSE — *Cross-Sectional Next-Day Return Prediction for Indian Equities*
> **Mode:** **PAPER ONLY.** No broker connection. No orders. No money. Ever.
> **North star:** *Given only information available at the close of day T, can we rank a fixed 50-stock NSE universe by the probability that each one closes green on day T+1?*
> **Brief version:** 1.0 · 15 September 2026

---

## 0. Read this first

You are building a **research instrument**, not a trading bot. The deliverable is a *trustworthy measurement*, not a profit.

Two questions live inside this project. Keep them separate in the code, in the metrics, and in every report you generate:

| | Question | Primary metric | Honest prior |
|---|---|---|---|
| **A** | Can the model **order** 50 stocks by next-day return better than chance? | Daily rank IC | Plausible — weak signal likely exists |
| **B** | Would acting on that ordering have **made money** after Indian transaction costs? | Net return vs. equal-weight benchmark | Unlikely — see §8 |

**A system can pass A and fail B. That is the expected outcome and it is a valid result.** Build the system so that outcome is visible rather than hidden.

### The single most important thing

The code in this project is easy. **Being correct is not.** Every look-ahead bug is a one-line mistake that produces a beautiful, completely false result — fitting a scaler before the split, a feature touching today's close, a forward-filled price, a shuffled time-series split. They are easy to write and nearly impossible to catch by reading.

Therefore: **build the leakage test suite (§13) BEFORE you build any model.** This is a hard ordering constraint, not a suggestion. Phase 6 gates Phase 7.

### If a result looks great, it is a bug

If at any point you observe a rank IC above ~0.06, a directional accuracy above ~60%, or a backtest Sharpe above 2.0, **stop and assume you have a leak.** Run the full §13 suite, report the numbers, and do not proceed until you can explain the result. Published cross-sectional daily models achieve IC in the 0.02–0.05 range. Anything dramatically better is a bug, not an edge.

---

## 1. Scope

### You ARE building

- A reproducible data pipeline over ~10 years of daily NSE OHLCV for a fixed 50-stock universe
- A point-in-time feature engine with enforced information cut-offs
- A walk-forward backtest harness with a realistic Indian transaction-cost model
- A ladder of models from random baseline up to gradient-boosted learning-to-rank
- An automated leakage-detection test suite wired into CI
- A metrics library with proper significance testing
- An HTML report generator
- A nightly forward-prediction logger (predictions only — **no orders**)

### You are NOT building — and must not

| Forbidden | Why |
|---|---|
| **Any broker API integration** (Zerodha Kite Connect or otherwise) | Paper only. No account, no credentials, no orders, no exceptions. |
| Any code that places, modifies or cancels an order | Same. |
| A live-trading runner, WebSocket feed, or order reconciliation layer | Out of scope entirely. |
| Deep learning (LSTM, GRU, Transformer, GNN) | Effective sample size is far smaller than the row count suggests. Tree models are the evidence-backed default for tabular panels. |
| Reinforcement learning | Requires a validated simulator you do not have. Fastest known route to overfitting. |
| A news / sentiment / LLM pipeline **backtested on historical news** | Historical news timestamps are unreliable and LLMs leak through their weights. Actively misleading. Excluded. |
| A fundamentals pipeline (P/E, ROE, debt) | Cannot plausibly predict a one-day return. Weeks of work for noise. |
| Kelly sizing, leverage, margin products | Amplifies an edge that has not been demonstrated. |
| Hyperparameter mega-sweeps | Every configuration inflates the multiple-testing penalty. Use the small fixed grid in §10. |
| A web dashboard / Streamlit app | The HTML report covers it. Do not build a UI. |

If you believe one of these is necessary, **stop and report** rather than building it.

---

## 2. Hard invariants

These are architectural laws. Violating any of them invalidates the project.

1. **Every raw-data read requires an explicit `cutoff` timestamp.** The accessor signature must make omitting it a `TypeError`, not a silent default. Look-ahead must be impossible to write by accident.
2. **Feature building is a pure function** of `(cutoff, store)`. No network calls, no global state, no mutation, no randomness without a seed.
3. **Truncation invariance holds.** Features computed on data truncated at date `T` must be *bitwise identical*, for all dates ≤ `T`, to features computed on the full dataset. This is the strongest single leakage test — see §13.2.
4. **Every transformation is fitted inside the fold.** Scalers, imputers, cross-sectional normalisation statistics, probability calibrators. Never on the full sample.
5. **No split ever shuffles time, and no date appears on both sides of a split boundary.** Group by date. Make shuffling impossible in the splitter API.
6. **Every result is traceable** to a content-hashed data snapshot, a config hash, a git SHA, and a seed.
7. **One decision code path.** `strategy.decide(...)` is called identically by the backtest engine and the forward-prediction runner. There is never a second copy.
8. **Costs are never a single constant.** Always the itemised model in §8, read from config.
9. **Every configuration evaluated out-of-sample is appended to `research_log`.** The count feeds the Deflated Sharpe Ratio. If you cannot say how many configurations were tried, there is no result.
10. **The last 24 months of data are a locked holdout.** Touched exactly once, at the very end, and the touch is logged with a timestamp.

---

## 3. Environment and stack

```
Python 3.11+ (3.12 preferred), managed with uv
```

| Purpose | Library | Notes |
|---|---|---|
| Dataframes | `pandas`, `numpy` | Panel is small (~125k rows). Do not reach for Spark/Polars. |
| Analytical store | `duckdb`, `pyarrow` | SQL over Parquet, zero-ops |
| Models | `lightgbm`, `scikit-learn` | `catboost`, `xgboost` only at rung 6 |
| Statistics | `statsmodels`, `scipy`, `arch` | Newey–West via `cov_type='HAC'`; stationary bootstrap via `arch.bootstrap` |
| Interpretability | `shap` | `TreeExplainer` only |
| Data acquisition | `yfinance` (primary) | See §4.1 for fallbacks |
| CLI / config | `typer`, `pydantic-settings`, `pyyaml` | Config typos must be errors, not silent behaviour changes |
| Experiment tracking | `mlflow` (local file backend) | Optional but recommended |
| Testing | `pytest`, `hypothesis`, `freezegun` | |
| Quality | `ruff`, `mypy` (strict on `domain/` and `strategy/`) | |
| CI | GitHub Actions | Must run the leakage suite on every push |

**Do not use** `backtrader`, `zipline`, `vectorbt`, `qlib` as frameworks (none model Indian charges, settlement or the closing auction — you will fight them), or `TA-Lib` (a C dependency for indicators that are five lines of pandas each, with better control over as-of semantics).

---

## 4. Data layer

### 4.1 Acquisition

**Primary source: `yfinance`**, using NSE tickers with the `.NS` suffix. Fetch with `auto_adjust=False` so you receive both `Close` and `Adj Close`.

```python
df = yf.download(tickers, start="2015-01-01", end=TODAY,
                 auto_adjust=False, progress=False, group_by="ticker")
```

**Critical convention:**
- **Returns are computed from `Adj Close`** (adjusted for splits, bonuses and dividends). Using raw `Close` makes every ex-dividend drop look like a loss — a systematic negative bias in a long-only daily strategy.
- **Price levels, tick rounding and cost calculations use raw `Close`** (STT is levied on actual traded value, not adjusted value).
- Keep both columns. Never mix them.

**Fallbacks, in order:**
1. If `yfinance` is unreachable (some sandboxes and corporate networks block Yahoo Finance at the egress layer), **fail loudly with a clear message**. Do not silently produce an empty dataset and do not attempt to route around a network policy.
2. Support a `--from-csv <dir>` mode that loads a directory of per-symbol CSVs with columns `date,open,high,low,close,adj_close,volume`, so the user can supply data obtained elsewhere.
3. Optionally implement an NSE bhavcopy reader as a *validation cross-check only*, not as the primary path.

### 4.2 Snapshots and provenance

Every fetch writes an immutable dated snapshot:

```
data/snapshots/<YYYY-MM-DD>T<HHMMSS>/
    bars.parquet
    index.parquet
    manifest.json      # {snapshot_id, taken_at, source, content_sha256, row_count, symbols[]}
```

- Compute a SHA-256 over the sorted serialised frame and record it.
- **On every new snapshot, diff against the previous one.** Any change to a *historical* bar (a date already present in the older snapshot) must be logged to `data/snapshots/drift.log` and printed. This is both a reproducibility guard and a free corporate-action detector: retroactive adjustment shows up here.
- Backtests cite a `snapshot_id`. Re-running a backtest against the same snapshot must be bit-identical.

### 4.3 Access API

```python
class BarStore:
    def get_bars(self, symbols: list[str], *, cutoff: pd.Timestamp,
                 lookback_days: int | None = None) -> pd.DataFrame:
        """Returns bars with date <= cutoff. `cutoff` is keyword-only and REQUIRED."""
```

Make `cutoff` keyword-only with no default. A test must assert that calling `get_bars(syms)` raises `TypeError`.

### 4.4 Data validation report (mandatory, Phase 2)

Generate `reports/data_quality.html` containing:

- Trading-day calendar reconstructed from the data; count of days per year; flag any year with an anomalous count
- Per-symbol coverage: first date, last date, number of missing trading days, longest gap
- Suspicious prints: `|daily return| > 20%` not explained by a known split; zero-volume days; `high < low`; `close` outside `[low, high]`
- Split/bonus artefacts: unadjusted close ratio jumps near round numbers (2:1, 5:1, 10:1)
- A clear verdict line: **"N symbols pass, M symbols flagged, K symbols should be dropped."**

**Do not proceed past this phase with symbols that fail.** Drop them and record why in `config/universe.yaml`.

### 4.5 Missing-data policy

- A symbol with no bar on date `t` is **excluded from that day's cross-section**. It is never imputed and never forward-filled into a feature.
- A day where fewer than 40 of 50 symbols have data is **skipped entirely** and logged.
- Every feature row carries a `staleness_flag` if any input was older than expected.

---

## 5. Universe

Fixed, committed to `config/universe.yaml`, never changed mid-experiment. ~50 liquid NSE names across ≥10 sectors, max 8 per sector.

```yaml
# config/universe.yaml — verify every ticker resolves before committing
universe:
  financials:   [HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK, BAJFINANCE, HDFCLIFE, SBILIFE]
  it:           [TCS, INFY, HCLTECH, WIPRO, TECHM]
  energy:       [RELIANCE, ONGC, BPCL, IOC, GAIL]
  auto:         [MARUTI, TATAMOTORS, M&M, BAJAJ-AUTO, EICHERMOT, HEROMOTOCO]
  pharma:       [SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, AUROPHARMA]
  fmcg:         [HINDUNILVR, ITC, NESTLEIND, BRITANNIA, DABUR, TATACONSUM]
  metals:       [TATASTEEL, JSWSTEEL, HINDALCO, VEDL, COALINDIA]
  materials:    [ULTRACEMCO, GRASIM, AMBUJACEM]
  infra_power:  [LT, NTPC, POWERGRID, ADANIPORTS]
  consumer:     [TITAN, ASIANPAINT, BHARTIARTL]
benchmark: "^NSEI"          # NIFTY 50
suffix: ".NS"
```

**Sector proxies:** do *not* depend on sector index tickers (coverage on free sources is unreliable). Compute each sector's return as the **equal-weighted mean of that sector's own universe members**, excluding the stock being scored (leave-one-out) so a stock is never compared against a basket containing itself.

**Survivorship bias — state it, don't hide it.** These 50 are today's survivors. Every report must carry the line: *"Universe fixed as of <date>; backtest results are subject to survivorship bias of unquantified magnitude."* If time allows, add a rule-based point-in-time universe as a robustness check and report the gap.

---

## 6. Features

All stock-level features are **cross-sectionally ranked within each date**, then mapped to `[-1, 1]` via `2 * (rank - 1) / (n - 1) - 1`. Store both raw and ranked values; models consume ranked.

### 6.1 Tier 1 — build these, expect most of the signal here

| Group | Features |
|---|---|
| **Reversal / momentum** | `ret_1d`, `ret_2d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d`, `ret_120d`, `ret_240d` (simple cumulative from `adj_close`); `mom_12_1` = 240d return excluding last 20d |
| **Vol-scaled reversal** | `ret_1d_scaled` = `ret_1d / vol_20d`; `ret_5d_scaled` |
| **Volatility** | `vol_5d`, `vol_20d`, `vol_60d` (std of daily returns); `vol_ratio` = `vol_5d / vol_60d`; `atr_14_pct` = ATR(14)/close |
| **Volume** | `vol_anomaly` = `log(volume_t / median(volume, 20))`; `log_traded_value_20d`; `amihud_20d` = `mean(|ret| / traded_value, 20)` |
| **Range / position** | `pos_in_range_20d` = `(close − min(low,20)) / (max(high,20) − min(low,20))`; `dist_52w_high` = `close / max(high,252) − 1`; `hl_pct_20d` = mean of `(high−low)/close` |
| **Session decomposition** | `gap` = `open_t / close_{t−1} − 1`, standardised by `vol_20d`; `c2o_sum_5d`; `o2c_sum_5d` |
| **Market-relative** | `beta_60d` vs NIFTY; `resid_ret_1d`, `resid_ret_5d`, `resid_ret_20d` (residual from rolling 60d regression of stock return on NIFTY return); `rel_strength_5d` = `ret_5d − beta·nifty_ret_5d`; `corr_60d_nifty` |
| **Sector-relative** | `sector_rel_ret_1d`, `_5d`, `_20d` (stock return minus leave-one-out sector mean) |

### 6.2 Tier 2 — market context (one row per day, identical across stocks)

These cannot help *rank* stocks, but they interact with stock features and gate the abstention rule. Include as columns; expect low standalone importance.

`nifty_ret_1d`, `nifty_ret_5d`, `nifty_vol_20d`, `nifty_vs_200dma`, `breadth` (fraction of universe up today), `avg_pairwise_corr_60d` (mean off-diagonal of the 60-day correlation matrix — a good cheap regime proxy), `xs_dispersion` (cross-sectional std of today's returns).

Optionally `india_vix` if a reliable free series is available; if not, **omit rather than approximate**.

### 6.3 Rules

- **Target 30–40 features. Do not exceed 50.** With ~2,500 days and heavy cross-sectional correlation, the effective sample is far smaller than 125,000 rows.
- Every feature is registered in `features/registry.py` with `(name, fn, dependencies, lookback_days, version)`.
- A feature with lookback `L` produces `NaN` for the first `L` rows. Never backfill.
- Winsorise raw features at the 1st/99th cross-sectional percentile before ranking.

### 6.4 Feature selection procedure

1. Univariate rank IC per feature on the **training window only**, with Newey–West t-stat and Benjamini–Hochberg correction across features. Publish the table.
2. Per-year IC for each feature. A feature whose IC flips sign across sub-periods is noise regardless of its full-sample number.
3. SHAP importance per walk-forward fold; rank-correlate importance vectors between folds. Below ~0.5 means the model is fitting fold-specific noise — report this.
4. **Ablate, don't accumulate.** Start from the full Tier-1 set and remove groups, reporting ΔIC. Adding features one at a time to a growing set overfits the selection process itself.

---

## 7. Labels

Compute all three forward horizons; the primary target is the demeaned close-to-close return.

```python
fwd_c2c = adj_close[t+1] / adj_close[t] - 1        # overnight gap + full next session
fwd_c2o = open_adj[t+1]  / adj_close[t] - 1        # overnight gap only
fwd_o2c = adj_close[t+1] / open_adj[t+1] - 1       # next session only
```

Then, per date, across the universe:

| Target | Definition | Use |
|---|---|---|
| `y_raw` | `fwd_c2c` | Reported only. Dominated by the market factor. |
| **`y_demeaned`** | `fwd_c2c − mean(fwd_c2c)` across the day's cross-section | **PRIMARY TRAINING TARGET** |
| `y_resid` | residual of `fwd_c2c` on `beta_60d · nifty_fwd_c2c` | Variant; better for a universe with dispersed betas |
| `y_rank` | within-day rank of `fwd_c2c`, normalised to `[0, n−1]` | Target for the `lambdarank` head |
| `y_green` | `1[fwd_c2c > 0]` | Reported diagnostic only — **never the primary training target** |
| `y_beat` | `1[fwd_c2c > mean(fwd_c2c)]` | The meaningful binary label |

**Why `y_demeaned` and not `y_green`:** for a large-cap Indian equity, market and sector factors account for 50–75% of daily return variance, and the market component is essentially unpredictable. A classifier trained on `y_green` spends most of its capacity failing to predict the market and settles on the base rate. Demeaning removes the common factor analytically so the model only has to learn what is learnable.

Winsorise all continuous targets at ±5σ before training; use Huber loss for regression heads.

---

## 8. Transaction-cost model

`config/charges.yaml`, every rate dated with a source URL. **Never hard-code. Never use a single blended constant.**

```yaml
# Verified against https://zerodha.com/charges/ on 2026-08-30. RE-VERIFY BEFORE USE.
as_of: "2026-08-30"
source: "https://zerodha.com/charges/"
delivery_cnc:
  brokerage_pct: 0.0
  stt_buy_pct: 0.001          # 0.1%
  stt_sell_pct: 0.001         # 0.1%
  stamp_buy_pct: 0.00015      # 0.015%
  exchange_txn_pct: 0.0000307 # 0.00307% NSE, both legs
  sebi_pct: 0.000001          # Rs 10 per crore
  ipft_pct: 0.000001          # Rs 10 per crore (VERIFY)
  gst_pct: 0.18               # on (brokerage + exchange_txn + sebi)
  dp_charge_flat: 15.34       # Rs per scrip per SELL day, GST inclusive
intraday_mis:
  brokerage_per_order: {pct: 0.0003, cap: 20.0}   # min(0.03%, Rs 20) per executed order
  stt_buy_pct: 0.0
  stt_sell_pct: 0.00025       # 0.025% sell side only
  stamp_buy_pct: 0.00003      # 0.003%
  exchange_txn_pct: 0.0000307
  sebi_pct: 0.000001
  ipft_pct: 0.000001
  gst_pct: 0.18
  dp_charge_flat: 0.0
slippage:
  base_bps: 2.0
  impact_coef: 0.5            # bps = base + coef * daily_vol_bps * sqrt(order_value / adv_value)
  min_bps: 1.0
  high_vol_multiplier: 2.0    # applied when nifty_vol_20d is in its top decile
```

### 8.1 Reference values — your unit tests must reproduce these

For a **₹1,00,000** position, buy and sell at the same price, 5 bps slippage per leg:

| Component | Delivery (CNC) | Intraday (MIS) |
|---|---:|---:|
| Brokerage | ₹0.00 | ₹40.00 |
| STT | ₹200.00 | ₹25.00 |
| Stamp duty | ₹15.00 | ₹3.00 |
| Exchange txn | ₹6.14 | ₹6.14 |
| SEBI + IPFT | ₹0.40 | ₹0.40 |
| GST @18% | ₹1.18 | ₹8.38 |
| DP charge | ₹15.34 | ₹0.00 |
| **Statutory subtotal** | **₹238.06 (0.238%)** | **₹82.92 (0.083%)** |
| Slippage (2 legs × 5 bps) | ₹100.00 | ₹100.00 |
| **All-in round trip** | **≈ ₹338 (0.34%)** | **≈ ₹183 (0.18%)** |

Write `tests/test_costs.py` asserting these to the paisa. If your implementation disagrees, your implementation is wrong.

### 8.2 The break-even that governs the project

A large-cap NSE stock has a daily return σ of roughly 1.6–2.0%. For a net annualised Sharpe of 1.0 on a one-position-per-day strategy:

```
required net daily mean = 1.8% / sqrt(250) ≈ 0.114%
required GROSS daily alpha = 0.114% + 0.34% ≈ 0.45%  (45 bps per trade)
```

Published cross-sectional ML decile spreads run **5–15 bps per day gross**. The strategy needs three to nine times the best-documented edge, in a market with roughly four times the transaction tax of the US.

**Print this break-even number at the top of every report.** It is the context in which every other number should be read.

### 8.3 Minimum viable position size

The flat ₹15.34 DP charge is 0.31% of a ₹5,000 position and 0.015% of a ₹1,00,000 one. Below roughly ₹50,000 per position, depository fees alone consume any plausible edge. Use ₹1,00,000 per position as the simulation default and expose it as config.

---

## 9. Backtest engine

### 9.1 Loop

Expanding-window walk-forward over trading dates:

```
for T in trading_dates[min_train:]:
    if is_retrain_day(T):                      # monthly, first trading day
        train_data = all rows with available_time < T
        model = train(train_data, purged_cv_for_hyperparams)
        register(model, git_sha, config_hash, snapshot_id, seed)
    features = features.build(cutoff=T)         # pure, no network
    preds    = model.predict(features)
    intents  = strategy.decide(T, features, preds, portfolio, config)   # SHARED
    fills    = sim_broker.execute(intents, bars[T], costs, slippage)
    portfolio.apply(fills)
    log_everything(T)                           # all 50 symbols, not just traded
```

- **Minimum training window: 3 years.** Retrain monthly, predict daily.
- Also run a **2-year rolling window** variant — non-stationarity may favour forgetting.
- Hyperparameters selected by **purged k-fold with embargo** *inside the training window only*: 1-day purge, 5-day embargo, folds grouped by date.
- The output is not a return series — it is a **full replay** containing every feature vector, prediction, intent and fill, written to Parquet, so any later question is answerable without re-running.

### 9.2 Fill simulator

| Aspect | Rule |
|---|---|
| Reference price | The day's close (`Close`, raw). Document that on a real exchange this would be the closing-auction equilibrium price, which may differ. |
| Slippage | Formula in `config/charges.yaml`; applied against the reference price, adverse direction always |
| Fill probability | Market orders always fill. A symbol with zero volume that day does not fill — log it as a missed trade and report the missed-trade rate as a result. |
| Capital | Enforce available cash, integer share quantities, and **T+1 settlement** (sale proceeds available next day). Do not let the simulator spend money it does not have. |
| Corporate actions | Handled implicitly by using `Adj Close` for returns. Assert that no simulated daily return exceeds ±25% without a logged explanation. |
| Costs | Full itemised model, per leg, per symbol. DP charge applies **per scrip per sell day** — so selling 3 symbols costs 3 × ₹15.34. |

### 9.3 Strategy / decision layer

One function, shared by the backtest and the forward runner:

```python
def decide(asof, features, predictions, portfolio, config) -> list[Intent]
```

Selection rules (all configurable, all reported):

| Rule | Default | Note |
|---|---|---|
| `K` (positions held) | 3 | Top-1 is statistically the worst way to test a hypothesis; K=3 cuts idiosyncratic variance ~√3 |
| Sizing | Inverse-volatility, equal risk contribution | `notional_i ∝ 1 / vol_20d_i`, then clamped |
| Position cap | 40% of equity | |
| Liquidity cap | ≤ 10 bps of 20-day median traded value | |
| Sector constraint | ≤ 1 name per sector in the top-K | |
| Correlation constraint | Cluster the universe on a Ledoit–Wolf-shrunk 60-day correlation matrix; ≤ 1 pick per cluster | This is the correlation-aware selection requirement, done cheaply |
| **Abstention** | Skip the trade if `predicted_edge_bps < round_trip_cost_bps + buffer` | Economically motivated; the highest-value rule in the set |
| **Overlap-aware rebalance** | If a held name is still in tomorrow's top-K, **do not sell and rebuy it** | Cuts turnover 30–50% for ~20 lines. Given that costs bind, this is the single highest-return change available. |
| Hold period | 1 day baseline; also sweep 3, 5, 10 | See §12 |

Keep a **fixed-1-day-hold variant** as the scientific reference (label and trade coincide exactly) alongside the overlap-aware variant.

**No stop-loss, no take-profit in v1.** Intraday σ is comparable to the entire predicted edge; a stop mostly harvests noise, and a take-profit truncates the right tail of the very distribution you are trying to measure.

---

## 10. Model ladder

**Climb in order. You may not build rung `n+1` until rung `n` has been beaten out of sample, on the same folds, with the difference bootstrapped.** Log every rung to `research_log`.

| Rung | Model | What it tests | Expected |
|---:|---|---|---|
| **0** | Uniform random top-K, 1,000 simulations | The harness itself | **Do this first.** Gives the null distribution of every metric you will later quote. |
| **1** | Equal-weight 50, monthly rebalance | Whether selection adds anything over just being invested | Surprisingly hard to beat net of costs |
| **2** | Single-feature sorts: `ret_1d` (reversal, bottom-K) and `ret_20d` (momentum, top-K) | Whether the classic effects exist here at all | If reversal has no IC, **stop and re-examine the data** before modelling |
| **3** | Ridge regression on `y_demeaned`; logistic on `y_beat` | Whether a linear combination beats single features | This is your real baseline; often within noise of the best model |
| **4** | LightGBM regression on `y_demeaned`, Huber objective | Whether non-linearity and interactions add anything | The workhorse |
| **5** | LightGBM `lambdarank`, one query group per date, `y_rank` as label | **The ranking-vs-classification question** | Expect a modest but real Precision@K improvement |
| **6** | CatBoost + XGBoost for ensemble diversity | Whether ensembling helps; whether disagreement is a usable confidence signal | Small IC gain; the disagreement signal is the more interesting output |
| **7** | Probability calibration (isotonic + Platt) on a purged validation fold | Whether the probabilities mean anything | Required before any confidence threshold is trustworthy |
| **8** | Rank-average ensemble of rungs 3–6 | Whether averaging beats the best single model OOS | Usually yes, modestly, and more stable across regimes |

### 10.1 Fixed hyperparameter grid — do not expand it

```yaml
lightgbm:
  num_leaves:      [15, 31, 63]
  min_data_in_leaf:[20, 50, 100]
  learning_rate:   [0.02, 0.05]
  feature_fraction:[0.6, 0.8]
  bagging_fraction:[0.7]
  lambda_l2:       [1.0, 10.0]
  n_estimators:    [300, 600]        # with early stopping on the purged validation fold
ridge:
  alpha: [0.1, 1.0, 10.0, 100.0]
```

That is 72 LightGBM configurations. **Every one counts toward the multiple-testing penalty.** Log them.

### 10.2 Seeds

Run every configuration with **5 seeds** and report the spread. **If seed variance exceeds the difference between two models, the difference is not real** — say so explicitly in the report.

### 10.3 The headline experiment (H2)

Same features, same folds, same seeds, three heads:

- (a) per-stock logistic classifier on `y_green`
- (b) LightGBM regressor on `y_demeaned`
- (c) LightGBM `lambdarank` on `y_rank`

Compare on Precision@1, Precision@3, daily rank IC and NDCG@5, with a **paired stationary bootstrap over days** for the confidence interval on the *difference*. Report the table regardless of which wins — a negative result here is valuable and saves building the wrong thing.

### 10.4 Recommended production shape

Two heads on the same features: the **ranker decides *which*** stocks, the **regressor decides *whether*** the expected edge clears the cost threshold. About 40 extra lines, and it resolves the tension between ranking (which loses magnitude) and regression (which is noisier for ordering).

---

## 11. Metrics

Split into the two families from §0. **Report both, every run. Never let one stand in for the other.**

### 11.1 Family A — did the model predict?

| Metric | Definition |
|---|---|
| **Daily rank IC** | Spearman correlation between predicted score and realised `fwd_c2c`, computed **within each date**, then averaged over dates. **This is the headline metric.** |
| ICIR | `mean(IC) / std(IC)` |
| **Newey–West t-stat on IC** | OLS of the daily IC series on a constant with `cov_type='HAC', maxlags=5`. **Significance bar is t > 3.0**, not 2.0 (Harvey–Liu–Zhu). |
| Precision@1, @3, @5 | Two versions: fraction of top-K that were green, **and** fraction that beat the day's cross-sectional mean |
| **Lift@K** | `Precision@K(green) − that day's fraction of green stocks`. **Raw accuracy without this is meaningless** — on a strong up day, random picking scores 90%. |
| Pesaran–Timmermann | Nonparametric test of directional predictive accuracy (formula in §11.3) |
| NDCG@5, MRR | Ranking metrics with return as gain |
| Brier score + reliability curve | Calibration of the probability head, decile-binned |
| **Long–short decile spread** | Mean return of top decile minus bottom decile, gross. **The purest measure of whether the model knows anything**, even though only the long side is tradeable. |
| R²_oos | Against a **zero** forecast, not against the mean (the Gu–Kelly–Xiu convention). Expect a very small positive number if things work. |

### 11.2 Family B — would it have made money?

Total return, CAGR, daily return distribution, Sharpe, Sortino, max drawdown, drawdown duration, volatility, win rate, average win / average loss, profit factor, turnover, **total costs as a percentage of gross P&L** (if this exceeds ~50%, the design is wrong regardless of the net result), alpha vs NIFTY, realised-vs-modelled slippage, and:

**Deflated Sharpe Ratio — mandatory on every reported Sharpe.**

```
Given N trials with observed per-period Sharpe ratios, let V = Var(SR across trials),
γ = 0.5772156649 (Euler–Mascheroni), Z⁻¹ = inverse standard normal CDF, e = Euler's number.

SR₀ = sqrt(V) · [ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

DSR = Φ(  (SR̂ − SR₀)·sqrt(T − 1)
        / sqrt(1 − γ₃·SR̂ + ((γ₄ − 1)/4)·SR̂²)  )

  SR̂ = observed per-period (NOT annualised) Sharpe
  γ₃ = skewness of the return series
  γ₄ = kurtosis (non-excess)
  T  = number of return observations
  N  = number of configurations tried  ← read this from research_log, honestly
```

SR̂ and SR₀ must be in the same per-period units. A DSR below 0.95 means the observed Sharpe is not distinguishable from the best you would expect by chance given how many things you tried.

### 11.3 Pesaran–Timmermann

```
P   = fraction of correct sign predictions
P_y = fraction of actual y > 0 ;  P_x = fraction of predicted > 0
P*  = P_y·P_x + (1 − P_y)(1 − P_x)
var(P)  = P*(1 − P*) / T
var(P*) = (1/T)(2P_y − 1)²·P_x(1 − P_x)
        + (1/T)(2P_x − 1)²·P_y(1 − P_y)
        + (4/T²)·P_y·P_x·(1 − P_y)(1 − P_x)
PT  = (P − P*) / sqrt(var(P) − var(P*))   ~ N(0,1)
```

### 11.4 Confidence intervals

**Stationary bootstrap over days** (`arch.bootstrap.StationaryBootstrap`, expected block length 5–10), ≥10,000 resamples, for every headline number. Point estimates alone are not a result.

**Cross-sectional correlation matters.** 50 stocks on one day are not 50 independent observations. With an average pairwise correlation of ~0.3, the effective cross-sectional sample size is closer to 3. Reflect this in how you compute and describe standard errors — never quote a t-stat computed as if all 125,000 rows were independent.

---

## 12. Baselines and experiments

### 12.1 Baselines — run all on identical dates, universe and cost model

| ID | Baseline | Beating it proves |
|---|---|---|
| **B0** | Uniform random top-K, 1,000 sims | **Beating the mean is not enough — you must beat the 95th percentile** |
| **B1** | Buy-and-hold NIFTY | That active management adds anything after costs |
| **B2** | Equal-weight 50, monthly rebalance | That *selection* adds value beyond universe construction. **The toughest and most important baseline.** |
| **B3** | Always-long the highest-beta name | That you are not merely levering market exposure |
| **B4** | Top-K by 20-day return (momentum) | That the model beats the classic effect |
| **B5** | Bottom-K by 1-day return (reversal) | **The one that matters at a daily horizon** — the effect your model is most likely to be silently rediscovering |
| **B6** | Ridge on Tier-1 features | That non-linearity earns its keep |

Compare using a **stationary bootstrap on the difference**, plus White's Reality Check / Hansen's SPA across the whole family of variants tried. The relevant question is never *"is my best model good?"* but *"is my best model better than the best model I'd expect to find by chance among this many attempts?"*

### 12.2 Required sweeps

| Sweep | Values | Why |
|---|---|---|
| **Horizon** | C2C, C2O, O2C | Lou–Polk–Skouras: momentum accrues overnight, reversal intraday. If Indian equities show the same split, close-to-close blends a predictable component with an opposing one. Costs an afternoon; potentially the most informative result in the project. |
| **Hold period** | 1, 3, 5, 10 days | Blitz et al.: short-horizon ML has large gross and ~zero net alpha; longer horizons amortise cost. **The most likely route from net-negative to net-positive.** |
| **K** | 1, 3, 5, 10 | Edge decay vs. diversification vs. per-scrip DP fees |
| **Cost multiplier** | 0×, 0.5×, 1×, 1.5×, 2×, 3× | Plot net Sharpe against the multiplier. **A strategy that dies between 1× and 1.5× is not a strategy — it is a measurement of your cost assumptions.** |
| **Product** | CNC (delivery) vs MIS (intraday) | ~0.34% vs ~0.18% round trip |
| **Training window** | Expanding vs 2-year rolling | Non-stationarity may favour forgetting |

### 12.3 Sub-period stability

Report IC and net Sharpe **per calendar year** and **per volatility tercile**. A strategy that worked only in 2020–21 has told you something about 2020–21.

---

## 13. The leakage test suite — BUILD THIS BEFORE ANY MODEL

`tests/test_leakage.py`, wired into CI. **Phase 6 gates Phase 7.** These six tests are the difference between a result and a story.

### 13.1 Shuffled-label placebo

Permute `y` **within each date** (preserving the cross-sectional structure) and re-run the entire pipeline end to end.

**Assert:** `|mean daily IC| < 0.005` and `|NW t-stat| < 2.0`, and the strategy's Sharpe falls inside the B0 random distribution.
**If it fails, you have a leak.** Do not proceed.

### 13.2 Truncation invariance ★ *the strongest test*

Build features on the dataset truncated at date `T`. Build features again on the full dataset. **Assert the two frames are identical, to floating-point equality, for every row with date ≤ `T`.**

Run it for 10 randomly chosen values of `T` spread across the sample. Any feature that peeks at future data — a full-sample scaler, a centred rolling window, a `bfill`, a groupby over all time — breaks this immediately.

### 13.3 Positive control (time-shift)

Inject `y` itself as a feature. **Assert the measured IC > 0.5.** This proves the harness can *detect* a signal — a placebo test alone does not.

### 13.4 Synthetic-signal injection ★ *rarely done, very valuable*

Add a synthetic feature `x = a·y + ε` with `a` chosen so the theoretical rank IC ≈ 0.10.

**Assert the pipeline recovers IC in `[0.07, 0.13]`** and that simulated P&L is of the right order of magnitude. This validates that the harness measures signal *of the correct size* — catching scaling, alignment and sign bugs that a binary pass/fail cannot.

### 13.5 Cut-off enforcement

**Assert** `BarStore.get_bars(symbols)` without `cutoff` raises `TypeError`. **Assert** that every feature function declares its `lookback_days` and that no feature reads a row with `date > cutoff` (instrument the store in test mode and record every date touched).

### 13.6 Split integrity

**Assert** no date appears in both the train and validation side of any fold; **assert** purge and embargo gaps are respected; **assert** that every fitted transformer (scaler, imputer, calibrator) was fitted on training rows only — check by comparing fitted statistics against a recomputation on the training subset.

### 13.7 Golden-file test

Freeze one day's feature matrix and portfolio state. Assert `strategy.decide` produces byte-identical intents. Any refactor that silently changes decisions then fails CI.

---

## 14. Reports

Single self-contained HTML per run at `reports/<run_id>.html`, no external assets. It must open with the **verdict block**:

```
GREENCLOSE run 2026-09-20T14:02  ·  model v0.4.1-lgbm-rank  ·  snapshot a91f3c…  ·  git 4b2e9d1
BREAK-EVEN CONTEXT: this strategy needs ~45 bps gross daily alpha to reach net Sharpe 1.0.

Mean daily rank IC      0.0184   (NW t = 2.31, 95% CI [0.004, 0.033])      ✗ below t>3 bar
Precision@3 lift        +1.9 pp  (95% CI [-0.4, +4.2])                     ✗ CI includes 0
Net return vs B2        -3.4% p.a. (95% CI [-7.1, +0.3])                   ✗
Costs / gross P&L       68%                                               ✗ above 50% threshold
Deflated Sharpe Ratio   0.41     (N = 214 configurations tried)            ✗
Leakage suite           6/6 PASS                                           ✓
VERDICT: Signal present but not significant. Not tradeable at daily turnover.
```

Then: per-year IC table, IC time series with confidence band, calibration curve, feature-importance stability across folds, the baseline comparison table with bootstrap CIs, all sweep results, the cost-multiplier curve, and a link to the replay Parquet.

**Write the verdict honestly and mechanically from the thresholds.** Do not soften it in prose.

---

## 15. Forward prediction log — the "green close" daily runner

**Predictions only. No orders. No broker. No money.**

A script `scripts/daily_forward.py` that, run any evening after market close:

1. Fetches the day's bars
2. Builds features at `cutoff = today 15:15 IST`
3. Loads the **pinned, frozen model artefact** and scores all 50 stocks
4. Appends one row **per stock** (all 50, not just the top-K) to `forward_log.parquet`:
   `date, symbol, model_version, score, rank, prob_green, prob_beat_mean, expected_return_bps, selected, abstained, abstain_reason`
5. On the next run, fills in realised outcomes for the previous date: `fwd_c2c, fwd_c2o, fwd_o2c, excess_vs_universe, direction_correct, beat_universe_mean`, plus simulated gross/net P&L using the cost model
6. Appends a one-line summary to `forward_log_summary.md`

**Why this matters:** it is the only evaluation that *provably cannot leak* — the prediction is timestamped and committed before the outcome exists. It costs nothing, needs no broker, and starts accumulating value the day it is switched on. Recommend turning it on in week 1 and never switching it off.

**Statistical honesty:** roughly 180 forward days are needed to establish a rank IC of 0.03 at t = 2; roughly 1,300 days for a P&L edge of 0.10%/day. A short forward log is an integrity check, not evidence of an edge — label it as such in the summary.

---

## 16. Repository structure

```
greenclose/
├── pyproject.toml                # uv-managed, uv.lock committed
├── README.md
├── config/
│   ├── base.yaml                 # universe ref, model, risk, sweep params
│   ├── charges.yaml              # every rate, dated, with source URL
│   ├── universe.yaml             # the frozen 50 + sector map
│   └── calendar.yaml             # NSE holidays, session times
├── src/greenclose/
│   ├── domain/
│   │   ├── models.py             # Bar, Features, Prediction, Intent, Fill, Position
│   │   └── protocols.py          # BarStore, Broker, Clock  (typing.Protocol)
│   ├── ingest/
│   │   ├── yahoo.py  csv_source.py  snapshots.py  universe.py
│   ├── features/
│   │   ├── registry.py           # name -> (fn, deps, lookback, version)
│   │   ├── price.py  market.py  context.py
│   │   └── build.py              # build(cutoff, store) -> DataFrame   [PURE]
│   ├── labels/
│   │   └── targets.py            # y_raw, y_demeaned, y_resid, y_rank, y_green, y_beat
│   ├── models/
│   │   ├── splitters.py          # purged/embargoed, group-by-date, cannot shuffle
│   │   ├── train.py  infer.py  calibrate.py  registry.py
│   ├── strategy/
│   │   └── decide.py             # THE shared decision function
│   │   └── selection.py  sizing.py  gates.py
│   ├── backtest/
│   │   ├── engine.py  sim_broker.py  costs.py
│   ├── evaluation/
│   │   ├── metrics.py            # IC, Precision@K, lift, NDCG, PT, Brier
│   │   ├── significance.py       # NW, stationary bootstrap, DSR, SPA
│   │   ├── baselines.py  report.py
│   ├── ops/
│   │   └── logging.py  research_log.py
│   └── cli.py                    # typer: fetch|validate|features|backtest|sweep|report|forward
├── scripts/
│   └── daily_forward.py
├── notebooks/                    # exploration only; never imported by src/
├── tests/
│   ├── test_leakage.py           # §13 — the six tests
│   ├── test_costs.py             # §8.1 reference values
│   ├── test_pointintime.py
│   ├── test_splitters.py
│   ├── test_decide.py            # golden file
│   └── test_metrics.py           # known-answer tests for IC, PT, DSR
├── data/                         # gitignored
│   ├── snapshots/  parquet/  models/
└── reports/                      # gitignored except .gitkeep
```

### CI (`.github/workflows/ci.yml`)

Runs on every push: `ruff` → `mypy` (strict on `domain/`, `strategy/`) → `pytest tests/` including the **full leakage suite on a small synthetic fixture dataset** (fast, no network). The leakage suite must be able to run without internet — generate a deterministic synthetic panel in `tests/fixtures/`.

---

## 17. Build order and acceptance gates

Each phase ends with something demonstrable. **Do not start a phase until the previous phase's gate passes.** Report the gate result before moving on.

| Phase | Build | Gate — must be true to proceed | Est. |
|---|---|---|---|
| **P0** | Repo scaffold, `pyproject.toml`, config files, CI skeleton, typed domain objects and protocols | `uv sync` works; `pytest` runs (0 tests); `ruff` and `mypy` clean | 0.5 d |
| **P1** | Data acquisition, snapshot store with content hashing + drift diffing, `BarStore` with required `cutoff` | 10 years of bars for ≥48 of 50 symbols stored; re-running the fetch produces an identical hash; `get_bars` without `cutoff` raises `TypeError` | 1 d |
| **P2** | Data validation report | `reports/data_quality.html` exists and **you have read it**; failing symbols dropped and recorded; verdict line printed | 0.5 d |
| **P3** | Feature engine + registry + **truncation-invariance test (§13.2)** | All Tier-1 + Tier-2 features build; §13.2 passes for 10 random cut-offs; feature count ≤ 50 | 1 d |
| **P4** | Labels + cost model | `tests/test_costs.py` reproduces the §8.1 table to the paisa; all six label variants computed | 0.5 d |
| **P5** | Walk-forward harness, sim broker, metrics library, baselines B0–B5 | B0 random distribution generated; B5 reversal baseline produces a plausible gross and net number; **report the net-of-cost result for B5 — this is the first genuinely informative output of the project** | 1 d |
| **P6** | **Full leakage suite in CI** | **All six tests in §13 pass.** This gate is absolute. | 0.5 d |
| **P7** | Model ladder rungs 0–8 | Each rung beaten by the next, or documented as not beaten; 5 seeds each; every config in `research_log` | 1.5 d |
| **P8** | Sweeps (§12.2) + H2 ranking-vs-classification experiment | All sweep tables produced with bootstrap CIs | 1 d |
| **P9** | Report generator + forward prediction runner | `reports/<run_id>.html` renders with a mechanically generated verdict; `daily_forward.py` writes and back-fills correctly | 0.5 d |
| **P10** | Locked holdout evaluation — **once only** | Touched once, logged with timestamp; result reported whether or not it agrees with the walk-forward | 0.5 d |

**Total ≈ 8 working days.**

### Minimum viable path (the 2-day spike)

If time is constrained, this subset produces a real first answer: **P0 (minimal) → P1 → P2 (abbreviated) → P3 (Tier-1 only) → P4 → P5 → P6 → P7 rungs 0–5.** Skip P8–P10.

A 2-day result is a **screen, not evidence** — it will be missing the point-in-time universe, the locked holdout, honest trial counting and full sub-period analysis. Label it that way in the report.

---

## 18. Definition of done

The project is complete when **all** of the following are true:

- [ ] `uv sync && pytest` passes from a clean clone, with the leakage suite green
- [ ] A single command reproduces the headline result end to end from a snapshot ID
- [ ] Re-running the same command against the same snapshot produces **bit-identical** output
- [ ] `reports/<run_id>.html` contains the verdict block, all Family A and Family B metrics with bootstrap CIs, all baselines, all sweeps, and the cost-multiplier curve
- [ ] `research_log` contains one row per out-of-sample configuration evaluated, and the DSR was computed using that count
- [ ] The locked holdout was evaluated exactly once and the result reported
- [ ] `daily_forward.py` has run successfully on at least 3 consecutive trading days and back-filled outcomes correctly
- [ ] `README.md` states, in the first paragraph, the honest headline finding — including if the finding is "no detectable edge"
- [ ] **Zero broker code, zero order-placement code, zero credentials anywhere in the repository**

---

## 19. Red flags — stop and report if you see these

| Observation | Almost certainly means |
|---|---|
| Rank IC > 0.06 | Leakage. Run §13 in full. |
| Directional accuracy > 60% | You are measuring against the wrong base rate, or leaking. Compute Lift@K. |
| Backtest Sharpe > 2.0 | Leakage, or costs not applied, or survivorship in the universe. |
| Placebo test shows non-zero IC | Confirmed leak. Do not proceed. |
| Truncation-invariance test fails | A feature peeks at the future. Bisect the feature registry. |
| Synthetic-signal test recovers the wrong magnitude | Scaling, alignment or sign bug in the pipeline. |
| Costs / gross P&L < 10% at daily turnover | Costs are not being applied correctly. Check the DP charge is per scrip per sell day. |
| Results change when you re-download data | Retroactive corporate-action adjustment. Check the snapshot drift log. |
| A model beats the reversal baseline by a huge margin | It has probably rediscovered reversal *and* something leaky. Check feature importance. |
| Feature importance reshuffles completely between folds | Fitting fold-specific noise. Report it; do not tune around it. |

**The general rule: this project's failure mode is a result that is too good, not one that is too bad.** Treat good news as a bug report until proven otherwise.

---

## 20. Reference reading

Consult these when a design decision is unclear. They are the evidence behind the choices in this brief.

### Directly analogous — the template

- **Krauss, Do & Huck (2017)**, *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500*, EJOR 259(2) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0377221716308657) · [open copy](https://hal.science/hal-01515120). **This is almost exactly our problem.** Daily, cross-sectional, predict outperformance vs. the cross-sectional median, long top-10 / short bottom-10. Uses *only* lagged multi-horizon returns. Note that its profitability decays sharply after 2001 — treat that as our base rate.
- **Feng et al. (2019)**, *Temporal Relational Ranking for Stock Prediction*, ACM TOIS — [arXiv](https://arxiv.org/abs/1809.09441) · [code](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking). The citation for ranking over classification. Its own ablations suggest the **ranking loss matters more than the network architecture** — which is why we use LightGBM `lambdarank` and not their model.

### Why we expect to fail question B

- **Blitz, Hanauer, Hoogteijling & Howard (2023)**, *The Term Structure of Machine Learning Alpha* — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4474637). **The paper most likely to describe our outcome.** Short-horizon ML has large gross alpha and ~zero net alpha post-2004. Prescription: longer horizons, lower turnover — hence the hold-period sweep in §12.2.
- **Avramov, Cheng & Metzker (2023)**, *Machine Learning vs. Economic Restrictions*, Management Science 69(5) — [link](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2022.4449). ML profitability concentrates in stocks that are expensive or impossible to trade. Our universe is liquid large caps by construction — the hardest place for ML alpha to survive.
- **Novy-Marx & Velikov (2016)**, *A Taxonomy of Anomalies and Their Trading Costs*, RFS 29(1) — [RFS](https://academic.oup.com/rfs/article-abstract/29/1/104/1844518). High-turnover strategies lose most of their gross alpha to costs; simple banding recovers a lot. The justification for overlap-aware rebalancing.

### Where the signal comes from at a one-day horizon

- **Jegadeesh (1990)**, *Evidence of Predictable Behavior of Security Returns*, JF 45(3) — [PDF](http://m.e-m-h.org/Jega90.pdf). Short-term reversal — the dominant documented daily cross-sectional effect, and the baseline our model must beat. Also the most cost-sensitive anomaly in existence.
- **Lou, Polk & Skouras (2019)**, *A tug of war: Overnight versus intraday expected returns*, JFE 134(1) — [PDF](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf). Momentum accrues overnight, reversal intraday. **The reason for the three-horizon design.** Two columns of arithmetic to test.
- **Agarwalla, Jacob & Varma**, *Fama-French and Momentum Factors for India* — [free data library](https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/). Free daily Indian factor returns. Use for risk attribution: regressing strategy returns on these tells you whether you have alpha or repackaged momentum.

### Statistical honesty

- **Bailey & López de Prado (2014)**, *The Deflated Sharpe Ratio* — [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf). Formula in §11.2. ~40 lines of NumPy. Mandatory.
- **Harvey, Liu & Zhu (2016)**, *… and the Cross-Section of Expected Returns*, RFS 29(1) — [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF). The **t > 3.0** bar.
- **White (2000)**, *A Reality Check for Data Snooping*, Econometrica 68(5) — [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152). Bootstrap comparison across a family of tried strategies.
- **Pesaran & Timmermann (1992)**, *A Simple Nonparametric Test of Predictive Performance*, JBES 10(4) — [JBES](https://www.tandfonline.com/doi/abs/10.1080/07350015.1992.10509922). The correct significance test for a directional claim.

### Model-class evidence

- **Grinsztajn, Oyallon & Varoquaux (2022)**, *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS — [arXiv](https://arxiv.org/abs/2207.08815). **Why LightGBM is the default and deep learning is out of scope.**
- **Zeng et al. (2023)**, *Are Transformers Effective for Time Series Forecasting?*, AAAI Oral — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26317) · [code](https://github.com/cure-lab/LTSF-Linear). A one-layer linear model beats published transformer forecasters. Do not reach for a time-series transformer.
- **Gu, Kelly & Xiu (2020)**, *Empirical Asset Pricing via Machine Learning*, RFS 33(5) — [NBER PDF](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf). Read for *method* — expanding-window training, strict OOS, R²_oos against zero. Ignore its feature count: it is monthly, with thousands of stocks and decades of data.

### Useful resources

- **[microsoft/qlib](https://github.com/microsoft/qlib)** — mine the **Alpha158 feature definitions** for well-tested price/volume feature formulas. Do **not** adopt Qlib as a framework.
- **[OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection)** — precise, peer-reviewed definitions of 200+ cross-sectional predictors. US data, but the definitions port and prevent subtly wrong feature implementations.
- **[shap/shap](https://github.com/shap/shap)** — `TreeExplainer` for the fold-stability diagnostic.

---

## 21. Context an implementer should know

Facts verified against primary sources on 30 August 2026. They shape the *interpretation* of results even though this is a paper-only build.

| Fact | Relevance here |
|---|---|
| **NSE Closing Auction Session went live 3 Aug 2026** for F&O-eligible stocks. Continuous trading ends **15:15**; an auction runs 15:20–15:30 with a ±3% band; the auction equilibrium price is the **official close**. | Our information cut-off is conceptually **15:15**, not 15:30. The simulator uses the daily close as a proxy for the auction price — **document this as a known approximation** in every report. |
| **STT is 0.1% on both legs** for delivery. | 20 bps of friction before anything else. The whole reason question B is hard. |
| **DP charge is ₹15.34 flat per scrip, per sell day.** | Scales with K, not with capital. Sets a minimum viable position size of ~₹50,000. |
| Kite historical prices are **retroactively adjusted** for corporate actions. | Free sources behave similarly. This is why snapshots are content-hashed and diffed. |
| Kite Connect has **no sandbox**; orders need a **registered static IP** from 1 Apr 2026; the access token expires at **06:00 IST daily**. | Purely context for why this project is paper-only. **Do not implement any of it.** |
| **30 trading days cannot detect a P&L edge.** At σ = 1.8%/day the smallest detectable mean is 0.66%/day. | A previous version of this project included a 30-day live experiment. **It has been deliberately removed.** Do not reintroduce a fixed short live window as a validation step. |

---

## 22. Assumptions to make without asking

You are expected to proceed without blocking. If any of these is wrong, the user will correct it.

1. Data source is `yfinance`; if unreachable, fail loudly and offer `--from-csv`.
2. History starts **2015-01-01** (or the earliest common date across the universe), ends today.
3. Simulation capital is **₹10,00,000**, position size **₹1,00,000**, `K = 3`.
4. Product is **CNC (delivery)** for the primary run; MIS is a sweep variant.
5. Primary target is **`y_demeaned`** on the **C2C** horizon; the other horizons are sweeps.
6. Timezone is **Asia/Kolkata**; all stored timestamps are timezone-aware UTC.
7. Currency is INR throughout; no FX.
8. Reports are written to `reports/`; nothing is published anywhere.
9. Everything runs on a laptop. **No cloud, no VM, no Docker required.**

**Do ask** before: changing the frozen universe, adding a model class outside the ladder, adding a data source that requires credentials or payment, or anything that would touch a broker.

---

## 23. How to report back

At the end of every phase, report in this shape — short, numeric, and honest:

```
PHASE P5 COMPLETE
Gate: PASS
Built: walk-forward harness, sim broker, metrics library, baselines B0-B5
Key numbers:
  B5 (1-day reversal, K=3, CNC): gross +7.2% p.a. / net -4.1% p.a. / costs 158% of gross P&L
  B2 (equal-weight 50):          net +11.8% p.a.
  B0 (random, 1000 sims):        net mean -5.9% p.a., 95th pct +2.1% p.a.
Notes: reversal has real gross alpha here but daily turnover destroys it — consistent
       with Novy-Marx & Velikov. Hold-period sweep in P8 is now the key experiment.
Next: P6 leakage suite.
```

**Never report a headline number without its confidence interval and the baseline it is being compared against.** Never describe a result as "promising" — give the number and let it speak.

---

## 24. The closing instruction

The most valuable thing this project can produce is **not a profitable strategy — it is a trustworthy measurement.** Most attempts at this fail not because markets are unpredictable but because the measuring instrument is broken: leakage, unrealistic costs, base-rate confusion, quiet mid-experiment tinkering.

Two sentences to keep in view while building:

> *The model predicting stocks that go green and the strategy making money are different claims, and both need separate evidence.*
>
> *If you cannot say how many configurations you tried, you do not have a result.*

Build the instrument. Report what it reads.

---

*End of brief. Paper only — no broker, no orders, no money.*
