# The Green Close Experiment

**A research and engineering plan for testing next-day cross-sectional predictability on a 50-stock NSE universe, executed via Zerodha Kite Connect.**

> Companion to the published artifact of the same name. Generated 30 August 2026.
> Facts about Kite Connect, Zerodha charges, the NSE Closing Auction Session and SEBI's
> retail algo framework were verified against primary sources on that date and are cited
> inline. All of them change — re-verify before the live stage.

---

*Research & engineering plan · v0.1 · 30 August 2026*

# Can we predict which stocks *close green* tomorrow?

A staged plan for testing next-day cross-sectional predictability across a 50-stock NSE universe, executed through Zerodha Kite Connect. Written to find out whether the edge is real — not to produce a backtest that looks profitable.

**Universe** ~50 NSE equities, F&O-eligible **Horizon** 1 trading day **Stages** Backtest → shadow → small live **Primary metric** Rank IC, not P&L

## 00 · The bottom line, before you write any code

*Read this section even if you read nothing else. It contains the two numbers that decide whether the rest of the plan is worth executing.*

Your request contains two different projects, and conflating them is the single most common way this kind of experiment fools its author. Keep them separate in your head, in your code, and in your logs:

### Project A — the prediction question

Given only information genuinely available at the day-*T* cut-off, can a model order 50 stocks by their day-*T+1* return better than chance?

Measured by **rank information coefficient**, Precision@K against the correct base rate, and calibration. Costs are irrelevant here. This question is *answerable* with the data and time you have.

### Project B — the trading question

Does acting on those predictions produce money after brokerage, STT, stamp duty, GST, DP charges, slippage and execution timing?

Measured by net P&L, net Sharpe, and alpha versus NIFTY. This question has a **much higher bar**, and the honest prior is that a daily-turnover single-stock strategy in Indian cash equities fails it.

A system can pass A and fail B. That is the most likely outcome, and it is still a scientifically useful result. Design the experiment so that outcome is visible rather than hidden.

### The cost wall

Indian equity delivery carries **Securities Transaction Tax at 0.1% on both the buy and the sell leg**. That is 20 basis points of pure friction before you have added stamp duty, exchange charges, GST, the flat depository charge on every sell, or a single paisa of slippage. Round-trip, on a ₹1,00,000 position:

| Component                   | Delivery (CNC), hold overnight      | ₹          | Intraday (MIS), same session    | ₹         |
|-----------------------------|-------------------------------------|------------|---------------------------------|-----------|
| Brokerage                   | Zero                                | 0          | min(0.03%, ₹20) × 2 legs        | 40.00     |
| STT                         | 0.1% buy + 0.1% sell                | 200.00     | 0.025% sell only                | 25.00     |
| Stamp duty                  | 0.015% buy side                     | 15.00      | 0.003% buy side                 | 3.00      |
| Exchange txn (NSE)          | 0.00307% × 2                        | 6.14       | 0.00307% × 2                    | 6.14      |
| SEBI + IPFT                 | ~₹10 + ₹10 per crore × 2            | 0.40       | same                            | 0.40      |
| GST @18%                    | on brokerage + txn + SEBI           | 1.18       | on brokerage + txn + SEBI       | 8.38      |
| DP charge                   | **₹15.34 flat per scrip, per sell** | 15.34      | Not applicable                  | 0         |
| **Statutory round trip**    | **≈ 0.238%**                        | **238.06** | **≈ 0.083%**                    | **82.92** |
| Slippage (assume 5 bps/leg) | market orders, liquid large-cap     | 100.00     | market orders, liquid large-cap | 100.00    |
| **All-in round trip**       | **≈ 0.34%**                         | **338**    | **≈ 0.18%**                     | **183**   |

<span class="small">Rates as published on [zerodha.com/charges](https://zerodha.com/charges/), fetched 30 Aug 2026. These change — read them from a config file, never hard-code them, and re-verify before the live stage.</span>

> **Skeptic's note — the arithmetic that should govern everything**
>
> A large-cap NSE stock has a daily return standard deviation around 1.6–2.0%. To earn a net annualised Sharpe of even 1.0 on a one-position-per-day strategy you need a net daily mean of roughly `1.8% ÷ √250 ≈ 0.114%`. Add back the 0.34% delivery round trip and your model must generate **~45 bps of gross daily alpha per trade**.
>
> For scale: cross-sectional machine-learning decile spreads in the published literature run roughly 5–15 bps *per day* gross, and [Blitz, Hanauer, Hoogteijling & Howard (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4474637) find that net-of-cost alpha from short-horizon ML models is close to zero after 2004 in US data. You are asking for three to nine times the gross edge that the best-documented models produce, in a market with roughly four times the transaction tax.
>
> This does not mean stop. It means: **state the break-even up front, measure against it, and do not let a pretty backtest talk you out of it.**

*\[Interactive break-even calculator — see the published page.\]*

### What this plan therefore recommends

1.  **Make the primary deliverable a prediction result, not a P&L result.** The 30-day live run is an *engineering validation*, not an edge test — see §13 for why 30 days cannot possibly establish statistical significance on returns.
2.  **Evaluate three horizons, not one** — close→close, close→open, open→close — because the cost structure and the signal structure differ enormously between them (§7, §10).
3.  **Prefer a cross-sectional ranking target over per-stock binary classification.** A binary "will it close green" classifier mostly learns to predict the market, which is close to unpredictable (§8).
4.  **Minimise turnover before you optimise the model.** Overlap-aware rebalancing, a cost-aware trade threshold and a longer hold are worth more than any model upgrade (§9, §A).
5.  **Build the leakage-detection harness before the model.** Placebo tests and a synthetic-signal injection test are the cheapest insurance available (§10, §27).

------------------------------------------------------------------------

# Part I · The research question

*What exactly is being claimed, what the literature says about it, and what the market microstructure will allow.*

## 01 · Problem definition

*Made precise enough that a reviewer could falsify it.*

### Formal statement

Let `U` be a fixed universe of `N ≈ 50` NSE-listed equities. Let `ℱ_T` be the information set genuinely observable at the day-*T* cut-off time `τ_T`. For each stock `i ∈ U` define the forward return `r_{i,T+1}` over a stated holding window.

We seek a scoring function `s(i, ℱ_T) → ℝ` such that the cross-sectional ordering induced by `s` is positively rank-correlated with the realised ordering of `r_{i,T+1}`, with the correlation statistically distinguishable from zero out of sample.

### Fixing the three things people usually leave vague

| Ambiguity                   | Decision for this project                                                                                                                      | Why                                                                                                                                                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| What is `τ_T`, the cut-off? | **15:15 IST** for F&O-eligible stocks; 15:30 for others. Features are frozen at that instant.                                                  | Continuous trading for F&O-eligible names now ends at 15:15 because of the Closing Auction Session (§12). Using 15:30 data for those names is look-ahead. |
| What is the holding window? | Three variants, evaluated separately: **C2C** (close *T* → close *T+1*), **C2O** (close *T* → open *T+1*), **O2C** (open *T+1* → close *T+1*). | They have different predictability and radically different costs. Collapsing them hides where any edge lives.                                             |
| What counts as "green"?     | Both the raw sign `r > 0` and the **cross-sectionally demeaned** sign `r − r̄_U > 0`.                                                           | On a day the whole market rises 1%, 45 of 50 stocks are "green" and a naive classifier scores 90% while knowing nothing.                                  |

### Scope boundaries

- **In scope:** daily-frequency, one decision cycle per day, long-only cash equity (plus optional intraday shorts for evaluation), fixed universe, single account.
- **Out of scope for v1:** intraday alpha, options, futures, pairs/stat-arb, market-making, anything requiring sub-second latency, anything requiring more than 10 orders per second.

> **Your "one buy phase, one sell phase" rule, formalised**
>
> Each trading day has exactly two decision windows: a **sell window** (exit positions opened previously) and a **buy window** (open new positions), each of which may touch multiple symbols. This maps cleanly onto a daily rebalance with an explicit ordering: compute target portfolio → diff against current portfolio → emit sells → wait for settlement of funds → emit buys. It is the right constraint: it caps turnover, makes the audit trail trivial, and keeps you far below any regulatory order-rate threshold.

## 02 · Research hypotheses

*Pre-registered, with the direction of the test and the specific statistic that decides it. Commit this to git with a timestamp before you look at any out-of-sample result.*

| ID   | Hypothesis                                                                        | Test statistic                                                | Pass threshold                                               |
|------|-----------------------------------------------------------------------------------|---------------------------------------------------------------|--------------------------------------------------------------|
| `H1` | Cross-sectional next-day ordering is predictable from EOD information.            | Mean daily Spearman rank IC, Newey–West t-stat                | IC \> 0.02, t \> 3.0 over ≥1000 OOS days                     |
| `H2` | Ranking beats independent per-stock binary classification.                        | Δ Precision@3 and Δ IC, paired bootstrap                      | 95% CI on the difference excludes 0                          |
| `H3` | Predictability survives realistic Indian transaction costs.                       | Net top-K minus equal-weight-50 spread; Deflated Sharpe Ratio | DSR \> 0 after correcting for number of configurations tried |
| `H4` | News/event features add incremental information beyond price and market features. | Nested model comparison on forward-collected news only        | ΔIC \> 0 with t \> 2 on ≥250 forward days                    |
| `H5` | Non-linear models beat regularised linear models on this panel.                   | OOS IC, gradient boosting vs ridge                            | ΔIC positive across ≥4 of 5 walk-forward folds               |
| `H6` | Live execution matches the simulator.                                             | Realised minus simulated fill, per order                      | Median absolute gap \< 8 bps over 30 days                    |

> **Null hypotheses you should genuinely expect to fail to reject**
>
> My prior, stated so you can hold me to it: `H1` is **plausible** (weak cross-sectional predictability from short-term reversal, volume and market-relative features is well documented). `H2` is **likely**. `H3` is **unlikely at daily turnover**. `H4` is **unlikely to be establishable** in under a year of forward-collected news. `H5` is a **coin flip** and often goes to the linear model on small panels.

### The falsification discipline

Keep a `research_log` table with one row per model/feature/hyperparameter configuration you evaluate out of sample. Every row increments the multiple-testing count that feeds the Deflated Sharpe Ratio ([Bailey & López de Prado, 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)). If you cannot state how many configurations you tried, you cannot claim significance. Additionally: hold out the most recent 24 months completely, touch it exactly once, and record the date you touched it.

------------------------------------------------------------------------

# Part II · The evidence base

*What the literature actually establishes about next-day cross-sectional prediction — including the findings that argue against this working. Deliverable items 18 and 19 sit here because they should shape the design, not follow it.*

## 18 · Research-paper bibliography

*Papers are grouped by the role they play in this project. Each entry ends with a blunt judgement about whether the methodology transfers to a 50-stock, one-day-horizon Indian equity problem.*

### A · Directly analogous: daily cross-sectional ML on a fixed stock universe

Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500 <span class="code-tag">reproducible</span>

Christopher Krauss, Xuan Anh Do, Nicolas Huck · 2017 · *European Journal of Operational Research* 259(2), 689–702 · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0377221716308657) · [open copy](https://hal.science/hal-01515120)

Core idea  
Each day, predict the probability that each S&P 500 constituent outperforms the cross-sectional median on the next day; go long the top 10 and short the bottom 10. This is **almost exactly your problem**, one horizon and one market removed.

Features  
Only lagged multi-horizon returns — cumulative returns over 1…20 days plus 40, 60, 120, 180, 240 days. Deliberately minimal; no news, no fundamentals.

Models  
Deep neural net, gradient-boosted trees, random forest, and an equal-weight ensemble of all three.

Dataset  
S&P 500 constituents, point-in-time membership, 1992–2015.

Horizon  
One trading day, close-to-close.

Results  
Daily returns around 0.45% before costs for the ensemble; directional accuracy only slightly above 50%; crucially, **the authors report that profitability decays sharply after 2001** and largely vanishes in the most recent sub-period.

Reproducible  
Fully — the feature set is trivially computable from Kite daily candles, and the ranking-by-median formulation ports directly.

Verdict  
**Use as the primary template.** Build this exact configuration as your first real model. Also treat its decay finding as the base rate for your own expectations.

Deep learning with long short-term memory networks for financial market predictions

Thomas Fischer, Christopher Krauss · 2018 · *European Journal of Operational Research* 270(2), 654–669 · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652) · [record](https://cris.fau.de/publications/208534319/)

Core idea  
Same experimental design as above, with an LSTM over a 240-day return sequence replacing the flat feature vector.

Features  
Sequence of standardised daily returns, 240 timesteps.

Models  
LSTM vs random forest, deep net, logistic regression.

Dataset  
S&P 500, 1992–2015.

Horizon  
One day.

Results  
LSTM beats the memory-free baselines, ~0.46% daily before costs; the edge again concentrates pre-2010 and the authors attribute much of it to short-term reversal that the LSTM rediscovers.

Reproducible  
Yes, but expensive relative to value on a 50-stock panel.

Verdict  
**Defer.** With 50 stocks × ~2,500 days you have ~125k samples with heavy cross-sectional correlation. That is not enough independent information to justify an LSTM before a tuned gradient-booster has been beaten.

Temporal Relational Ranking for Stock Prediction <span class="code-tag">code</span>

Fuli Feng, Xiangnan He, Xiang Wang, Cheng Luo, Yiqun Liu, Tat-Seng Chua · 2019 · *ACM TOIS* 37(2) · [arXiv](https://arxiv.org/abs/1809.09441) · [code](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)

Core idea  
Reformulate stock prediction as a **learning-to-rank** problem rather than classification or regression, and inject inter-stock relations (sector, supply chain, shared ownership) via a Relational Stock Ranking layer over an LSTM.

Features  
Normalised price/return sequences plus a stock relation graph built from Wikidata and sector membership.

Models  
LSTM + temporal attention + graph convolution; trained with a combined pointwise regression and pairwise ranking loss.

Dataset  
NASDAQ (1,026 stocks) and NYSE (1,737 stocks), 2013–2017; both released.

Horizon  
One day.

Results  
Higher Mean Reciprocal Rank and back-tested return than classification baselines; the ranking loss contributes more than the graph component.

Reproducible  
Code and data are public. The *ranking loss idea* is trivially reproducible in LightGBM (`objective="lambdarank"`); the graph part needs a relation source you would have to build for Indian names.

Verdict  
**Adopt the ranking framing, skip the architecture.** This paper is your best citation for the "ranking vs. independent classification" question you raised — and its own ablations suggest the loss function matters more than the network.

Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach <span class="code-tag">code</span>

Ramit Sawhney, Shivam Agarwal, Arnav Wadhwa, Tyler Derr, Rajiv Ratn Shah · 2021 · *AAAI-21* · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/16127) · [code](https://github.com/midas-research/sthan-sr-aaai)

Core idea  
Model higher-order relations between stocks as a hypergraph (industry hyperedges, corporate-relation hyperedges) with temporal Hawkes attention, trained with a ranking objective to select top-K stocks.

Features  
Price sequences plus hypergraph structure over industry and corporate relations.

Models  
Temporal Hawkes attention + hypergraph attention; pairwise ranking loss.

Dataset  
NASDAQ, NYSE, TSE; public.

Horizon  
One day.

Results  
Beats RSR and other graph baselines on ranking metrics and simulated profit.

Reproducible  
Code is public; adapting the relation construction to NSE is real work with no obvious data source.

Verdict  
**Not for v1.** Its contribution is exactly the correlation-structure modelling you asked about, but on 50 stocks you can capture the same information with a rolling correlation matrix and sector dummies at 1% of the complexity.

### B · The canonical ML asset-pricing results — and the papers that puncture them

Empirical Asset Pricing via Machine Learning <span class="code-tag">replication available</span>

Shihao Gu, Bryan Kelly, Dacheng Xiu · 2020 · *Review of Financial Studies* 33(5), 2223–2273 · [RFS](https://academic.oup.com/rfs/article/33/5/2223/5758276) · [NBER PDF](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf) · [open replication](https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/)

Core idea  
The reference comparison of ML methods for the cross-section of expected returns; establishes that trees and neural nets beat linear models, mostly by capturing interactions and non-linearity in momentum, liquidity and volatility.

Features  
94 firm characteristics × 8 macro predictors, with interactions — 900+ predictors.

Models  
OLS, PCR, PLS, elastic net, GBRT, random forest, and neural nets of 1–5 layers.

Dataset  
~30,000 US stocks, 1957–2016, monthly.

Horizon  
**Monthly**, not daily.

Results  
Out-of-sample R² of ~0.4% monthly; a value-weighted decile long-short from the neural net earns ~2% per month gross.

Reproducible  
The method, yes. The 94-characteristic dataset needs CRSP/Compustat equivalents that do not exist cheaply for India.

Verdict  
**Read for method, do not copy the design.** Monthly horizon, thousands of stocks, decades of data. Your panel has 50 stocks and a one-day horizon: three orders of magnitude less cross-sectional information. Adopt its *discipline* (expanding-window training, strict OOS, R²<sub>oos</sub> against a zero forecast) and ignore its feature count.

Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability

Doron Avramov, Si Cheng, Lior Metzker · 2023 · *Management Science* 69(5), 2587–2619 · [Management Science](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2022.4449) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3450322)

Core idea  
Takes the ML-in-asset-pricing literature and asks what survives once you exclude microcaps, distressed and hard-to-arbitrage stocks, and apply realistic trading frictions.

Features  
Same characteristic sets as Gu–Kelly–Xiu.

Models  
Neural nets, GBRT, random forest, and the "conditional autoencoder" family.

Dataset  
US equities.

Horizon  
Monthly.

Results  
**ML profitability is heavily concentrated in stocks that are expensive or impossible to trade**, spikes during high-sentiment and high-limits-to-arbitrage periods, and shrinks sharply when economic restrictions are imposed.

Reproducible  
The *restriction methodology* is directly reproducible: rerun your own results excluding the least liquid third of the universe and see how much survives.

Verdict  
**Adopt as a mandatory robustness check.** Your universe is liquid large caps by construction, which is the hardest place for ML alpha to survive. That is the honest way round — but it means you should expect smaller numbers than the headline literature.

The Term Structure of Machine Learning Alpha

David Blitz, Matthias Hanauer, Tobias Hoogteijling, Clint Howard · 2023 · *Journal of Financial Data Science* · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4474637) · [summary](https://www.robeco.com/en-uk/insights/2023/07/the-term-structure-of-machine-learning-alpha)

Core idea  
ML models trained on short-horizon returns produce large *gross* alphas that are dominated by fast-decaying signals with punishing turnover. Models trained on longer-horizon targets produce lower gross but higher **net** alpha.

Features  
Standard characteristic set.

Models  
Gradient boosting and neural nets, trained on 1-month vs 3/6/12-month forward returns.

Dataset  
US equities, long sample.

Horizon  
1 to 12 months.

Results  
**Net-of-cost alpha for short-horizon ML models is close to zero after 2004.** Longer training horizons plus efficient trading rules restore significant net returns.

Reproducible  
The horizon-sweep experiment is cheap and you should run it: train the same model to predict 1-day, 3-day, 5-day and 10-day forward returns and compare net Sharpe.

Verdict  
**This is the paper most likely to describe your outcome.** Its prescription — lengthen the horizon, reduce turnover — is the single highest-value modification to your original design.

### C · Where the signal actually comes from at a one-day horizon

Evidence of Predictable Behavior of Security Returns

Narasimhan Jegadeesh · 1990 · *Journal of Finance* 45(3), 881–898 · [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1990.tb05110.x) · [PDF](http://m.e-m-h.org/Jega90.pdf)

Core idea  
The founding short-term reversal result: last period's relative losers outperform last period's relative winners at daily-to-monthly horizons.

Features  
Lagged own returns, cross-sectionally ranked.

Models  
Cross-sectional regressions and decile sorts.

Dataset  
NYSE/AMEX, 1934–1987.

Horizon  
Daily and monthly.

Results  
Highly significant negative first-order serial correlation in relative returns; a reversal strategy earned ~2% per month gross.

Reproducible  
Trivially. This is your first non-random baseline.

Verdict  
**Essential.** At a one-day horizon, short-term reversal is the dominant documented cross-sectional effect. If your ML model cannot beat a plain reversal sort, it has learned nothing. Note also that reversal is largely a liquidity-provision premium and is the *most* cost-sensitive anomaly in existence.

A tug of war: Overnight versus intraday expected returns

Dong Lou, Christopher Polk, Spyros Skouras · 2019 · *Journal of Financial Economics* 134(1), 192–213 · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650) · [PDF](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf)

Core idea  
Decomposing daily returns into the overnight and intraday components reveals that momentum accrues **entirely overnight** while reversal accrues **entirely intraday** — the two halves of the day are driven by different investor clienteles.

Features  
Close-to-open and open-to-close return decompositions.

Models  
Portfolio sorts and Fama–MacBeth regressions.

Dataset  
US equities, 1993–2013.

Horizon  
Sub-daily components of daily returns.

Results  
Persistent, economically large separation of signal by session.

Reproducible  
Completely — Kite gives you open and close, so the decomposition is two columns of arithmetic.

Verdict  
**Adopt immediately, and it is the reason for the three-horizon design in §01.** If Indian equities show the same split, then close-to-close blends a predictable component with an opposing one and destroys your own signal. Test it on day one; it costs an afternoon.

Four Factors in Indian Equity Market: Fama-French and Momentum factors for India <span class="code-tag">free data</span>

Sobhesh Kumar Agarwalla, Joshy Jacob, Jayanth R. Varma · 2013, updated continuously · IIM Ahmedabad · [data library](https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/) · [working paper](https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/four-factors-India-90s-onwards-IIM-WP-Version.pdf)

Core idea  
Constructs and freely publishes market, size, value and momentum factor returns for the Indian equity market, updated on an ongoing basis.

Features  
Daily and monthly factor return series for India.

Models  
Fama–French style factor construction adapted to Indian market conditions.

Dataset  
Indian listed equities from the 1990s onwards.

Horizon  
Daily factors available.

Results  
Documents that value and momentum premia exist in India; momentum is strong but with severe crash risk.

Reproducible  
**You do not need to reproduce it — you can download it.**

Verdict  
**Use as a free, India-specific feature source and as a risk-attribution benchmark.** Regressing your strategy's returns on these four factors is how you find out whether you have alpha or just repackaged momentum.

### D · News, text and language models

Which News Moves Stock Prices? A Textual Analysis

Jacob Boudoukh, Ronen Feldman, Shimon Kogan, Matthew Richardson · 2013 · NBER WP 18725 · [NBER](https://www.nber.org/papers/w18725) · [PDF](https://www.nber.org/system/files/working_papers/w18725/w18725.pdf)

Core idea  
Most "news" is noise. When you filter to *identified, relevant* news events using structured text analysis, the relationship between news and price moves becomes far stronger and the fraction of return variance explained by news rises substantially.

Features  
Event-type classification (earnings, M&A, analyst actions, legal, products) plus sentiment, from a commercial text engine.

Models  
Variance decomposition and event studies.

Dataset  
S&P 500 firms, Dow Jones newswire.

Horizon  
Daily.

Results  
Identified relevant news explains a much larger share of price moves than raw news counts or undifferentiated sentiment.

Reproducible  
The event-typing idea is reproducible with an LLM classifier; the newswire is not free.

Verdict  
**This justifies your instinct to do event extraction rather than sentiment scoring.** "Positive/negative" on an aggregator headline is close to useless; "Q1 EPS beat by 12%, guidance raised" is a feature.

FinBERT: Financial Sentiment Analysis with Pre-trained Language Models <span class="code-tag">code + weights</span>

Dogu Araci · 2019 · arXiv:1908.10063 · [arXiv](https://arxiv.org/abs/1908.10063) · [code](https://github.com/ProsusAI/finBERT) · [weights](https://huggingface.co/ProsusAI/finbert)

Core idea  
Domain-adapt BERT on financial text, then fine-tune for three-class sentiment.

Features  
Raw sentence text.

Models  
BERT-base, further pre-trained on a financial corpus, fine-tuned on Financial PhraseBank.

Dataset  
Financial PhraseBank (~4,800 sentences), FiQA.

Horizon  
Not a return-prediction paper — a classification paper.

Results  
State of the art on those benchmarks at the time; roughly 86% accuracy.

Reproducible  
Weights are one line from Hugging Face; runs on CPU for a few hundred headlines a day.

Verdict  
**Use it, but understand what it is.** FinBERT was trained on analyst-style English financial sentences. It has never seen an Indian business-news headline and it does not know that "SEBI issues show-cause notice" is bad. Validate it on a hand-labelled sample of 200 Indian headlines before you trust a single feature derived from it.

Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models

Alejandro Lopez-Lira, Yuehua Tang · 2023 · arXiv:2304.07619 · [arXiv](https://arxiv.org/abs/2304.07619) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788)

Core idea  
Prompt an LLM with a news headline and a company name, ask whether it is good or bad for the stock, and use the resulting score to sort stocks.

Features  
LLM sentiment score per headline, aggregated per stock-day.

Models  
GPT-family models vs traditional sentiment methods.

Dataset  
US stocks with headline data, Oct 2021 onward — chosen to lie beyond the model's training cutoff.

Horizon  
Next-day and next-few-day returns.

Results  
LLM scores predict next-day returns and beat dictionary methods; the effect is stronger in small caps and after costs shrinks considerably.

Reproducible  
Yes, and cheaply — a few hundred headlines a day is a trivial LLM bill.

Verdict  
**Worth testing in Phase 2, with one enormous caveat:** for any historical backtest, the LLM has probably seen the outcome. This is *look-ahead through model weights* and it is invisible in your data pipeline. LLM news features can only be honestly validated on news collected forward from today.

Predicting Returns with Text Data <span class="code-tag">replication</span>

Zheng Tracy Ke, Bryan Kelly, Dacheng Xiu · 2019 · NBER WP 26186 · [PDF](https://www.nber.org/system/files/working_papers/w26186/w26186.pdf) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3389884) · [community replication](https://github.com/mannymistry/predictreturnswithtext)

Core idea  
A supervised, interpretable alternative to black-box NLP: screen words by their correlation with returns, then score documents by a simple likelihood model (SESTM). No pre-trained language model needed.

Features  
Sentiment-charged word counts, learned from returns rather than from a labelled sentiment corpus.

Models  
Screening + topic model + MLE scoring.

Dataset  
Dow Jones Newswire, 1989–2017.

Horizon  
Daily.

Results  
Outperforms dictionary and off-the-shelf ML sentiment for return prediction; the learned word list looks nothing like a general sentiment lexicon.

Reproducible  
The algorithm is a few hundred lines and needs no GPU.

Verdict  
**The right first text model for you.** It learns what words matter *for Indian stock returns specifically*, sidesteps FinBERT's domain mismatch, and cannot leak through pre-trained weights. Its weakness — needing a lot of labelled-by-returns text — is real, so it belongs in Phase 2.

### E · Statistical honesty: the papers that stop you fooling yourself

The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality

David Bailey, Marcos López de Prado · 2014 · *Journal of Portfolio Management* 40(5) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) · [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

Core idea  
Given *N* strategy configurations tried, the expected maximum Sharpe ratio under the null of zero skill is substantially above zero. The DSR discounts an observed Sharpe by the number of trials and by skewness/kurtosis.

Reproducible  
~40 lines of NumPy.

Verdict  
**Mandatory.** Implement it in week one and report DSR alongside every Sharpe you ever quote. This requires you to honestly count trials — hence the `research_log` table in §14.

… and the Cross-Section of Expected Returns

Campbell Harvey, Yan Liu, Heqing Zhu · 2016 · *Review of Financial Studies* 29(1), 5–68 · [RFS](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824) · [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF)

Core idea  
After accounting for the hundreds of factors that have been tested, a newly claimed predictor needs a t-statistic above roughly **3.0**, not 2.0, to be credible.

Verdict  
**Adopt t \> 3.0 as your significance bar** for any claimed predictive result, and apply Benjamini–Hochberg across features when you screen them.

A Reality Check for Data Snooping

Halbert White · 2000 · *Econometrica* 68(5), 1097–1126 · [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152)

Core idea  
A bootstrap procedure that tests whether the best strategy among many has genuine predictive superiority over a benchmark, correcting for the search itself. Hansen's SPA test (2005) refines it.

Verdict  
**Use the stationary-bootstrap version** when comparing your ML strategy against the five baselines in §21. Comparing point estimates without it is meaningless when you have tried many models.

A Simple Nonparametric Test of Predictive Performance

M. Hashem Pesaran, Allan Timmermann · 1992 · *Journal of Business & Economic Statistics* 10(4) · [JBES](https://www.tandfonline.com/doi/abs/10.1080/07350015.1992.10509922)

Core idea  
A test of whether directional forecasts are independent of realised directions — the correct significance test for "does my up/down prediction beat chance", accounting for the base rate.

Verdict  
**This is the right test for your headline claim**, far more appropriate than quoting raw accuracy. Implement it; it is ten lines.

### F · Model-class evidence: what to build, and what not to

| Paper                                                                                                                                                                                                                                                                                    | What it establishes                                                                                                                                                                     | Consequence for this project                                                                                                                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Grinsztajn, Oyallon & Varoquaux (2022)**, "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS · [arXiv](https://arxiv.org/abs/2207.08815)                                                                                                              | On medium-sized heterogeneous tabular data, gradient-boosted trees beat tuned deep nets, mainly because of irregular target functions, uninformative features, and rotation invariance. | Your data is exactly this shape. **LightGBM/CatBoost is the right default**; a neural net must clear a high bar to justify itself.                                                                                                                                      |
| **Zeng, Chen, Zhang & Xu (2023)**, "Are Transformers Effective for Time Series Forecasting?", AAAI Oral · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26317) · [code](https://github.com/cure-lab/LTSF-Linear)                                                               | A one-layer linear model (DLinear) beats a series of published transformer forecasters on long-horizon benchmarks.                                                                      | Do not reach for a time-series transformer. If you want sequence modelling, **test a linear autoregressive baseline first** — it frequently wins.                                                                                                                       |
| **Kelly, Malamud & Zhou (2024)**, "The Virtue of Complexity in Return Prediction", *Journal of Finance* · [JF](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13298) · [replication](https://github.com/zivmi/voc_reproduction)                                                    | Heavily over-parameterised ridge-type models with random features can improve out-of-sample return prediction, contradicting the usual parsimony intuition.                             | An interesting late experiment (random-feature ridge is cheap), but the result is contested and sensitive to shrinkage choices. **\[defer\]**                                                                                                                           |
| **Chen, Pelger & Zhu (2024)**, "Deep Learning in Asset Pricing", *Management Science* · [arXiv](https://arxiv.org/abs/1904.00745) · [code](https://github.com/LouisChen1992/Deep_Learning_in_Asset_Pricing)                                                                              | Uses an adversarial (GAN-style) objective grounded in the no-arbitrage condition rather than plain return regression.                                                                   | Elegant, code available, but designed for monthly horizons and large panels. **\[out of scope\]**                                                                                                                                                                       |
| **Moody & Saffell (2001)**, "Learning to trade via direct reinforcement", *IEEE TNN* 12(4) · [ACM/IEEE](https://dl.acm.org/doi/abs/10.1109/72.935097); **FinRL** (Liu et al., 2021) · [arXiv](https://arxiv.org/abs/2111.09395) · [code](https://github.com/AI4Finance-Foundation/FinRL) | RL can optimise a differential Sharpe ratio directly including costs; FinRL packages modern deep-RL agents for trading.                                                                 | **Genuinely not relevant to v1.** RL needs an environment simulator that is at least as accurate as your backtester — you do not have one yet — and it is the easiest way in existence to overfit 2,500 days of data. Revisit only after §29 is passed. **\[not now\]** |

### G · Portfolio construction, sizing, costs and uncertainty

| Paper                                                                                                                                                                                                               | Core contribution                                                                                                                                                                                                                                   | Use here                                                                                                                                                                                   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Novy-Marx & Velikov (2016)**, "A Taxonomy of Anomalies and Their Trading Costs", *RFS* 29(1) · [RFS](https://academic.oup.com/rfs/article-abstract/29/1/104/1844518) · [NBER](https://www.nber.org/papers/w20721) | Classifies anomalies by turnover and shows that high-turnover strategies — precisely the short-horizon ones — lose most or all of their gross alpha to costs. Also shows that simple cost-mitigation (banding, patient rebalancing) recovers a lot. | **Read the cost-mitigation section carefully.** Buy/sell banding is the cheapest improvement available to you (§09).                                                                       |
| **López de Prado (2016)**, "Building Diversified Portfolios that Outperform Out of Sample", *JPM* 42(4) · [JPM](https://jpm.pm-research.com/content/42/4/59.short)                                                  | Hierarchical Risk Parity: cluster assets by correlation, allocate recursively down the dendrogram; avoids inverting an ill-conditioned covariance matrix.                                                                                           | The **clustering step alone** is what you want for correlation-aware top-K selection. You do not need the full allocation machinery for 1–3 positions.                                     |
| **Ledoit & Wolf (2004)**, "Honey, I Shrunk the Sample Covariance Matrix", *JPM* 30(4) · [PDF](http://www.ledoit.net/honey.pdf)                                                                                      | Shrinkage estimator for covariance matrices that is well-conditioned even when observations are scarce relative to assets.                                                                                                                          | Use `sklearn.covariance.LedoitWolf` for the 50×50 correlation matrix. With 50 assets and 60-day windows, the sample estimate is genuinely unusable.                                        |
| **MacLean, Thorp & Ziemba**, "Good and Bad Properties of the Kelly Criterion" · [PDF](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf)                                                          | Full Kelly maximises long-run growth but produces drawdowns most humans cannot tolerate, and is extremely sensitive to estimation error in the edge.                                                                                                | **Never use full Kelly here.** Your edge estimate will be wrong by more than the edge itself. Quarter-Kelly at most, and only after §29.                                                   |
| **Angelopoulos & Bates (2021)**, conformal prediction tutorial · [arXiv](https://arxiv.org/abs/2107.07511) · [code](https://github.com/aangelopoulos/conformal-prediction)                                          | Distribution-free prediction intervals with finite-sample coverage guarantees, from any model, using a calibration split.                                                                                                                           | Split conformal is ~20 lines and gives you an honest "how uncertain is this prediction" signal for the abstention rule. **\[phase 2, cheap\]**                                             |
| **Guo, Pleiss, Sun & Weinberger (2017)**, "On Calibration of Modern Neural Networks", ICML · [arXiv](https://arxiv.org/abs/1706.04599)                                                                              | Modern classifiers are systematically overconfident; temperature scaling on a held-out set fixes most of it.                                                                                                                                        | You asked about calibration — this plus isotonic regression is the answer. Report reliability diagrams and Brier score, and calibrate on a purged validation fold, never on training data. |
| **Lundberg & Lee (2017)**, "A Unified Approach to Interpreting Model Predictions", NeurIPS · [arXiv](https://arxiv.org/abs/1705.07874) · [code](https://github.com/shap/shap)                                       | Shapley-value attributions for individual predictions, with a fast exact algorithm for trees.                                                                                                                                                       | Use `TreeExplainer` for feature-importance stability checks across folds. If the top features reshuffle every fold, you are fitting noise — that is a *diagnostic*, not a nice-to-have.    |

> **What the literature does not contain**
>
> I could not find a credible, peer-reviewed study demonstrating a **net-of-cost profitable daily-rebalanced cross-sectional strategy on Indian large-cap equities**. The Indian ML-prediction literature that does exist is dominated by index-level (NIFTY 50 index, not constituents) direction papers with small samples, non-walk-forward validation, and accuracy figures that do not survive scrutiny. Treat any paper reporting \>70% next-day directional accuracy as almost certainly leaking — the usual culprits are scaling before splitting, shuffled cross-validation on time series, and using same-day close in the feature set.

## 19 · Papers and resources with runnable code

*Prioritised by how much of your build they actually remove. Column three is what you should copy, not what the repo does.*

| Resource                                                                                                                                  | What it gives you                                                                                                                                                                                  | Take this, ignore the rest                                                                                                                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [microsoft/qlib](https://github.com/microsoft/qlib)                                                                                       | A complete AI-for-quant platform: point-in-time data layer, Alpha158/Alpha360 feature sets, walk-forward "rolling" workflows, LightGBM/LSTM/GRU/Transformer baselines, and a portfolio backtester. | **The Alpha158 feature definitions** — a well-tested, published set of 158 price/volume features, which saves you weeks of feature engineering and gives you a citable baseline. Also its rolling-retrain design. *Do not* adopt Qlib as your framework: its data layer assumes its own binary format and you will fight it over Indian corporate actions. |
| [fulifeng/Temporal_Relational_Stock_Ranking](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)                               | Reference implementation of ranking losses for stock selection, plus released NASDAQ/NYSE datasets with relation graphs.                                                                           | The **pairwise ranking loss formulation** and their evaluation harness (MRR, IRR). Port the loss idea to LightGBM lambdarank rather than running their TensorFlow 1.x code.                                                                                                                                                                                |
| [midas-research/sthan-sr-aaai](https://github.com/midas-research/sthan-sr-aaai)                                                           | Hypergraph attention learning-to-rank for stock selection.                                                                                                                                         | Their **evaluation protocol** for top-K selection. The model itself is out of scope.                                                                                                                                                                                                                                                                       |
| [ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) · [weights](https://huggingface.co/ProsusAI/finbert)                              | Ready-to-run three-class financial sentiment.                                                                                                                                                      | Everything — it is a two-line dependency. Just validate on Indian headlines first.                                                                                                                                                                                                                                                                         |
| [aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction)                                               | Minimal, readable conformal prediction notebooks.                                                                                                                                                  | The split-conformal regression notebook, ~20 lines, for prediction intervals on expected return.                                                                                                                                                                                                                                                           |
| [shap/shap](https://github.com/shap/shap)                                                                                                 | TreeExplainer for LightGBM/XGBoost/CatBoost.                                                                                                                                                       | Per-fold feature attribution for the stability diagnostic.                                                                                                                                                                                                                                                                                                 |
| [OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection) · [site](https://www.openassetpricing.com/)                     | Chen & Zimmermann's open reproduction of 200+ published cross-sectional predictors, with code and downloadable portfolio returns.                                                                  | **The signal definitions.** A precise, peer-reviewed catalogue of how each characteristic is actually computed — invaluable for avoiding subtly wrong feature definitions. US data, but the definitions port.                                                                                                                                              |
| [IIM-A India factor library](https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/)                                                | Daily Indian market/size/value/momentum factor returns, free.                                                                                                                                      | Direct download. Your risk-attribution benchmark and a source of market-context features.                                                                                                                                                                                                                                                                  |
| [zerodha/pykiteconnect](https://github.com/zerodha/pykiteconnect)                                                                         | Official Python client for Kite Connect, including the `KiteTicker` WebSocket client with binary packet parsing.                                                                                   | Everything. Do not write your own HTTP or WebSocket layer.                                                                                                                                                                                                                                                                                                 |
| [yumoxu/stocknet-dataset](https://github.com/yumoxu/stocknet-dataset) (Xu & Cohen, ACL 2018, [paper](https://aclanthology.org/P18-1183/)) | Aligned tweets + prices for 88 stocks; the standard benchmark for text-based movement prediction.                                                                                                  | A **sanity-check corpus**: run your text pipeline on it and confirm you reproduce published accuracy before trusting it on your own scraped news.                                                                                                                                                                                                          |
| [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)                                                                           | DLinear/NLinear reference implementations.                                                                                                                                                         | The one-layer linear baseline. Use it to keep any sequence model honest.                                                                                                                                                                                                                                                                                   |

------------------------------------------------------------------------

# Part III · The market you are actually trading in

*Broker capabilities, regulatory constraints and a microstructure change that landed four weeks ago and invalidates the obvious execution design.*

## 12 · Zerodha / Kite Connect integration plan

*Everything below is from the official documentation and Zerodha support pages, fetched 30 August 2026. Rates and rules change; the integration layer must read them from config.*

> **The change that reshapes the whole design**
>
> NSE and BSE launched the **Closing Auction Session (CAS) on 3 August 2026** for all cash-segment stocks that have derivatives available — which is where essentially every liquid, diversified 50-stock universe lives. For those stocks:
>
> - Continuous trading now ends at **15:15**, not 15:30.
> - 15:15–15:20 is a transition window with **no orders accepted**.
> - 15:20–15:25 accepts market and limit orders; 15:25–15:30 accepts **limit orders only**, with a random close between 15:28 and 15:30.
> - Orders must be within **±3% of the reference price** (the 15:00–15:15 VWAP) or they are cancelled.
> - Stop-loss and iceberg orders are **excluded** from the auction; GTT and conditional orders **do not trigger** after 15:15.
> - The **equilibrium price of the auction is the official close** — matching at 15:30–15:35 — replacing the old last-30-minute VWAP.
> - MIS intraday auto-square-off for CAS stocks moves to **15:12**.
>
> Sources: [Zerodha Z-Connect on CAS](https://zerodha.com/z-connect/general/everything-you-need-to-know-about-closing-auction-session-cas), [Zerodha support](https://support.zerodha.com/category/trading-and-markets/trading-faqs/general/articles/closing-auction-session).

**Design consequences.** Your information cut-off `τ_T` is 15:15, so the whole decision pipeline — fetch, feature-build, predict, size, emit orders — has to complete inside roughly four minutes if you want to trade in Order Entry Session I. That is comfortable for 50 stocks but only if you have pre-warmed everything except the final bar. Second: because the official close is now an auction equilibrium and MIS squares off at 15:12, an intraday strategy can no longer exit "at the close" — it exits before the auction even begins. Third: sending a market order into Session I is the closest available analogue to "trade at the close", and it will fill at the equilibrium price.

### Access, cost and the personal-use tier

| Item                                | Current position                                                                                        | Implication                                                                                                |
|-------------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Order/portfolio APIs                | **Free** for personal use (announced April 2025)                                                        | Execution costs nothing to integrate.                                                                      |
| Live WebSocket + historical candles | **₹500/month** per API key (reduced from ₹2,000 in May 2025); historical data included in the paid plan | Budget ₹500/mo from day one — you need historical candles for the backtest and for daily feature building. |
| Sandbox / paper environment         | **Does not exist.** Zerodha states plainly that there is no API sandbox.                                | You must build your own simulated broker. This is not optional — see §11.                                  |
| Data redistribution                 | Displaying or redistributing Kite data externally violates exchange data-vending policy                 | Keep dashboards private. Do not publish a live price feed.                                                 |

<span class="small">Sources: [Kite Connect pricing](https://zerodha.com/products/api/), [free personal APIs announcement](https://zerodha.com/z-connect/updates/free-personal-apis-from-kite-connect), [API sandbox FAQ](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/api-sandbox), [Kite Connect FAQs](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/kite-connect-api-faqs).</span>

### Regulatory constraints on retail algos

> **Static IP is mandatory from 1 April 2026 — this constrains your deployment**
>
> Under SEBI's retail algo framework and the NSE operating procedure, **API order placement is only accepted from a registered static IP**. Up to two IPs (one primary, one optional secondary) can be registered on the developer dashboard, and **they can only be changed once per calendar week**. Orders from unregistered IPs are rejected. Market-data and other endpoints are not IP-restricted.
>
> Other requirements that touch your code directly:
>
> - **10 orders/second cap** per trading account unless the algo is registered with the exchange and given a unique algo ID. You will use ~6 orders per day, so this is a non-issue — but stay well under it and never build retry logic that could burst.
> - **Market orders require `market_protection`**. Orders with market protection set to 0 are rejected. Set it explicitly on every market order.
> - Order slicing must not exceed 10 slices.
>
> Sources: [Zerodha static IP article](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/static-ip), [Kite Connect compliance thread](https://kite.trade/forum/discussion/15912/preparing-to-comply-with-sebis-retail-algo-rules-static-ip-ratelimits-order-types), [NSE framework overview](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework).

### API surface: what you use, and how

| Capability                    | Endpoint / mechanism                                                                                                                                                                                                                                                                                                     | How this project uses it                                                                                                                                                                                                                                                                                                                                                     |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Authentication**            | Redirect to `kite.zerodha.com/connect/login?v=3&api_key=…` → receive `request_token` at your redirect URL → `POST /session/token` with `checksum = SHA256(api_key + request_token + api_secret)` → receive `access_token`.                                                                                               | **The access token expires at 06:00 IST the next day — a regulatory requirement with no workaround.** This means a human login step every trading day. Plan for it: a small local login helper that opens the URL, captures the redirect, exchanges the token and writes it to your secret store. Budget 60 seconds of your morning; do not build a credential-scraping bot. |
| **Instrument identification** | `GET /instruments` — gzipped CSV, regenerated daily, with `instrument_token`, `tradingsymbol`, `exchange`, `segment`, `lot_size`, `tick_size`.                                                                                                                                                                           | Download once per day *before* anything else and snapshot it. Symbols change on corporate actions; `instrument_token` is the stable key for WebSocket and historical calls. Never key your database on `tradingsymbol` alone.                                                                                                                                                |
| **Historical OHLCV**          | `GET /instruments/historical/:token/:interval` with `from`/`to`. Intervals: minute, 3/5/10/15/30/60-minute, day. Max window per request: **60 days (minute), 100 (3–10 min), 200 (15–30 min), 400 (60 min), 2000 (day)**. Daily candles reach back to the late 1990s for many NSE names; minute data starts around 2015. | Backfill once into your own store, then fetch only incrementals. Rate limit is 3 req/s — a full 50-stock, 10-year daily backfill is a few dozen calls. **Prices are adjusted for corporate actions retroactively**, which is a leakage hazard: see §04.                                                                                                                      |
| **Live market data**          | WebSocket `wss://ws.kite.trade` with `api_key` + `access_token`; modes `ltp` (8 B), `quote` (44 B), `full` (184 B, with 5-level depth). Max **3,000 instruments per connection, 3 connections per API key**. Binary packets.                                                                                             | You need one connection, ~51 instruments, `quote` mode, for the last hour of the session. Use `KiteTicker` from pykiteconnect. Also subscribe to order postbacks (JSON text frames on the same socket) for near-real-time fill confirmation.                                                                                                                                 |
| **Snapshot quotes**           | `GET /quote` (max 500 instruments, 1 req/s), `/quote/ohlc` and `/quote/ltp` (max 1000).                                                                                                                                                                                                                                  | Fallback if the WebSocket drops, and for the 15:14 feature freeze. Note the 1 req/s limit on full quotes — one call covers all 50 symbols, so this is fine.                                                                                                                                                                                                                  |
| **Order placement**           | `POST /orders/:variety` — varieties `regular`, `amo`, `co`, `iceberg`, `auction`. Products `CNC`, `MIS`, `NRML`, `MTF`. Types MARKET, LIMIT, SL, SL-M. Validity DAY, IOC, TTL. Supports a free-text `tag`.                                                                                                               | Use `regular` + `CNC` (delivery) or `MIS` (intraday). **Always set the `tag`** to your internal decision ID — it is the join key between your database and Zerodha's records and it survives into the order book. Always set `market_protection`.                                                                                                                            |
| **Order status verification** | `GET /orders` (day's orders), `GET /orders/:order_id` (full status history), `GET /orders/:order_id/trades` (individual fills). Statuses run PUT ORDER REQ RECEIVED → VALIDATION PENDING → OPEN PENDING → OPEN → COMPLETE / REJECTED / CANCELLED.                                                                        | **Never trust the response to the place call as confirmation of a fill.** Poll `/orders/:id` until terminal, and reconcile fills from `/trades`, which carries the actual executed prices and timestamps you need for the experiment log.                                                                                                                                    |
| **Portfolio & positions**     | `GET /portfolio/holdings` (with `t1_quantity` for unsettled shares), `GET /portfolio/positions` (`net` and `day`).                                                                                                                                                                                                       | Reconcile against your own ledger every morning and every evening. A mismatch is a stop-trading event, not a warning.                                                                                                                                                                                                                                                        |
| **Funds**                     | `GET /user/margins/equity`                                                                                                                                                                                                                                                                                               | Check before sizing. Never assume the sell proceeds are available for the same day's buys — under T+1 they are usually usable but confirm against the actual available balance.                                                                                                                                                                                              |
| **Errors**                    | `TokenException` (403, session dead), `InputException` (400), `MarginException`, `HoldingException`, `OrderException`, `NetworkException` (OMS unreachable), `DataException`, plus HTTP 429 for rate limits and 502/503/504 for OMS problems.                                                                            | Distinguish *retryable* (429, 502, 503, 504, NetworkException) from *terminal* (InputException, MarginException, HoldingException). Retry the first class with exponential backoff and jitter, capped at 3 attempts; **never blind-retry a place-order call** without first checking the order book for a duplicate.                                                         |

<span class="small">Endpoint details from [Kite Connect v3 documentation](https://kite.trade/docs/connect/v3/) — [auth](https://kite.trade/docs/connect/v3/user/), [historical](https://kite.trade/docs/connect/v3/historical/), [orders](https://kite.trade/docs/connect/v3/orders/), [WebSocket](https://kite.trade/docs/connect/v3/websocket/), [portfolio](https://kite.trade/docs/connect/v3/portfolio/), [exceptions](https://kite.trade/docs/connect/v3/exceptions/).</span>

### Rate limits, in one place

**1/s** — Quote endpoint

**3/s** — Historical candles

**10/s** — Orders & everything else

**400/min** — Order cap per minute

**5,000/day** — Order cap per day

**25** — Modifications per order

Your daily usage is roughly 55 historical/quote calls and 6 order calls. You are using about 0.1% of your allowance. Build a token-bucket limiter anyway — the failure mode you are protecting against is a retry loop during an outage, not normal operation.

### Settlement and product choice — the decision that dominates your cost model

| Approach                            | Mechanics                                                                          | Round-trip cost | Captures                                        | Verdict                                                                                                                                                    |
|-------------------------------------|------------------------------------------------------------------------------------|-----------------|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CNC, close→close**                | Buy in CAS on day *T*; shares are T1 on *T+1*; sell in CAS on *T+1* (a BTST sale). | ~0.34%          | Overnight gap + full next session               | Matches the research question most literally, but carries the highest cost *and* short-delivery/auction risk on the BTST leg if the counterparty defaults. |
| **CNC, close→close, hold 3–5 days** | Same, but only rebalance the tail of the portfolio.                                | ~0.07–0.11%/day | Same, amortised                                 | **\[recommended\]** Cost per day falls with holding period. This is the Blitz et al. prescription applied.                                                 |
| **MIS, open→close same day**        | Buy at/after open on *T+1*, square off before 15:12.                               | ~0.18%          | Intraday session only; misses the overnight gap | Cheapest per round trip and permits shorting, but you are predicting a different (and per Lou–Polk–Skouras, possibly opposite-signed) quantity.            |
| **CNC, close→open**                 | Buy in CAS on *T*, sell at open on *T+1*.                                          | ~0.34%          | Overnight gap only                              | Highest cost against the smallest return window. Interesting for research, not for trading.                                                                |

> **Practical hazards nobody mentions until they bite**
>
> - **BTST short delivery.** Selling T1 shares that were never delivered to you puts you in the exchange auction with a penalty. Rare in liquid large caps, but budget for it and log it.
> - **Short selling is intraday-only** in Indian cash equity. Any "sell" leg of a long-short design must be MIS and squared off the same session.
> - **±3% auction band.** A limit order outside the band is cancelled silently from your perspective. Always verify order status after CAS.
> - **Circuit limits and trading halts** can make a symbol untradeable. Your decision layer must handle "the stock I picked cannot be bought today".
> - **Freeze quantity** limits per-order size on some symbols; large orders must be sliced (≤10 slices).
> - **The post-close 15:50–16:00 session**, where market orders execute at the closing price, historically existed for cash equity. Its status for CAS stocks under the new regime is *ambiguous in the sources I could reach*. Verify empirically with a one-share order before you design around it.

------------------------------------------------------------------------

# Part IV · System design

*Modules, data, features and news. The guiding constraint: this is a small-data problem wearing a big-data costume — 50 stocks × 2,500 days is 125,000 rows. Build accordingly.*

## 03 · System architecture

*Modular, but deliberately unfashionable. No message bus, no orchestration platform, no feature-store product.*

> **The one architectural decision that matters most**
>
> There must be **exactly one implementation of the decision logic**, called identically by the backtester, the paper trader and the live trader:
>
>     def decide(asof: datetime, features: pd.DataFrame,
>                portfolio: Portfolio, config: Config) -> list[Intent]
>
> The three runners differ only in where `features` comes from and where `Intent` objects go: a fill simulator, a shadow logger, or the Kite API. Every research-vs-live divergence bug you will otherwise spend weeks on comes from having two copies of this function.

### Module map

| Layer          | Modules                                      | Responsibility                                                                            | Runs when                         |
|----------------|----------------------------------------------|-------------------------------------------------------------------------------------------|-----------------------------------|
| **Ingestion**  | `ingest.instruments`                         | Daily instrument dump → snapshot table; symbol/token mapping with validity ranges         | 08:30 daily                       |
|                | `ingest.bars`                                | Historical & incremental OHLCV for universe + indices; corporate-action drift detection   | backfill once, then 15:16 & 16:30 |
|                | `ingest.live`                                | WebSocket ticker; last-price cache; order postbacks                                       | 14:30–15:40                       |
|                | `ingest.context`                             | Index levels, India VIX, USD/INR, crude, gold, US index closes, GIFT Nifty, IIM-A factors | 15:16 daily                       |
| **News**       | `news.collect`                               | Poll sources; store raw with `first_seen_at`                                              | every 15 min                      |
|                | `news.dedupe`                                | URL canonicalisation + MinHash near-duplicate clustering                                  | on ingest                         |
|                | `news.link`                                  | Entity resolution to universe symbols; relevance scoring                                  | on ingest                         |
|                | `news.score`                                 | Event-type classification + sentiment + novelty; model-versioned                          | on ingest                         |
| **Features**   | `features.build`                             | Pure functions: (as-of cut-off, raw store) → feature frame. No side effects, no network.  | 15:14 live; batch in research     |
|                | `features.registry`                          | Feature definitions, versions, and the leakage contract for each                          | —                                 |
| **Modelling**  | `model.train`                                | Walk-forward training, purged CV, calibration, artefact registration                      | weekly / on schedule              |
|                | `model.infer`                                | Load pinned artefact, score the panel, emit ranked predictions + uncertainty              | 15:15                             |
|                | `model.registry`                             | Model artefacts keyed by git SHA + config hash + data snapshot hash                       | —                                 |
| **Decision**   | `strategy.decide`                            | The single shared function above: ranking → gates → selection → sizing → intents          | 15:16                             |
|                | `risk.gates`                                 | Pre-trade checks: exposure, concentration, correlation, liquidity, kill switches          | 15:16                             |
|                | `risk.sizing`                                | Volatility-scaled sizing with hard caps                                                   | 15:16                             |
| **Execution**  | `exec.broker`                                | Kite adapter behind a `Broker` protocol — the only module that imports `kiteconnect`      | 15:20                             |
|                | `exec.sim`                                   | Fill simulator implementing the same `Broker` protocol                                    | backtest/paper                    |
|                | `exec.reconcile`                             | Order → trade → position reconciliation; ledger truth                                     | 15:36 & 16:30                     |
| **Evaluation** | `backtest.walkforward`                       | Rolling train/predict/trade loop over history                                             | research                          |
|                | `eval.metrics`                               | IC, Precision@K, calibration, Sharpe, DSR, PT test, bootstrap CIs                         | research + daily                  |
|                | `eval.report`                                | Daily experiment log row + human-readable report                                          | 17:00                             |
| **Ops**        | `ops.alerts`, `ops.health`, `ops.killswitch` | Telegram alerts, heartbeats, manual halt file                                             | continuous                        |

### Daily timeline (F&O-eligible universe)

| IST   | Step                                                                                             | Failure behaviour                                                     |
|-------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| 08:30 | Human login → access token; instrument dump; health check; reconcile holdings vs ledger          | No token by 09:00 → alert, skip the day                               |
| 09:15 | (If intraday variant) place entry orders for yesterday's decisions                               | —                                                                     |
| 14:30 | WebSocket up; warm feature cache with everything except today's final bar                        | Fall back to REST quotes                                              |
| 15:12 | Freeze news window; compute all news/context features                                            | Missing news → set news features to their "no news" sentinel, flag it |
| 15:14 | Snapshot quotes; build the day-*T* bar; run `features.build(asof=15:15)`                         | Any stale symbol → exclude from candidate set, log                    |
| 15:15 | Inference → ranked predictions with probabilities and uncertainty; persist *before* any decision | Model load failure → abstain for the day                              |
| 15:16 | `strategy.decide` → intents; risk gates; sizing; persist decisions                               | Gate breach → abstain, alert                                          |
| 15:18 | Emit **sell** intents first (exits)                                                              | Reject → alert, retry once, then manual                               |
| 15:20 | Emit **buy** intents into CAS Order Entry Session I, with market protection                      | Same                                                                  |
| 15:36 | Poll order status until terminal; pull trades; reconcile                                         | Unreconciled after 10 min → alert loudly                              |
| 16:30 | End-of-day bars; realised returns; experiment-log row; daily report                              | —                                                                     |

## 04 · Data architecture

*Point-in-time correctness is the entire game. Everything else in this section serves it.*

### The as-of contract

Every row of every raw table carries three timestamps, and no exception is permitted:

| Column           | Meaning                                    | Example — a quarterly result      |
|------------------|--------------------------------------------|-----------------------------------|
| `event_time`     | When the thing happened in the world       | Quarter ended 30 Jun              |
| `available_time` | The earliest moment we could have known it | Filed with exchange 24 Jul, 18:42 |
| `ingested_at`    | When our system actually stored it         | 24 Jul, 18:47                     |

Feature building takes a `knowledge_cutoff` and filters `available_time <= cutoff`. One function, applied uniformly, kills the large majority of look-ahead bugs. Make it impossible to bypass: the raw-data accessor should *require* a cutoff argument, so that forgetting it is a `TypeError` rather than a silent 40% jump in backtest Sharpe.

> **Three India-specific leakage traps**
>
> **1. Retroactively adjusted prices.** Kite's historical candles are adjusted for bonuses, splits, rights and extraordinary dividends — retroactively. The series you download today is *not* what you would have seen on day *T*. Returns stay correct, but any feature touching price *levels* or raw volume is distorted, and re-downloading silently changes your past. Mitigation: store an **immutable dated snapshot** of every backfill with a content hash, diff each new snapshot against the last, and alert on any change to a historical bar. That diff is also a free corporate-action detector.
>
> **2. Close price ≠ official close.** A daily candle's close is the last traded price. The exchange's official close is now the CAS equilibrium price for F&O stocks (and previously the 15:00–15:30 VWAP). If your strategy trades "at the close", you must know which of these you are simulating. **Validation test:** reconcile Kite daily closes against NSE bhavcopy for 50 symbols × 60 days and quantify the difference before you trust a single close-to-close backtest number.
>
> **3. Fundamentals dated by period, not by filing.** Almost every free Indian fundamentals source stamps a row with the quarter it describes, not the day it was published. Using that date gives your model results roughly three to six weeks before the market had them. If you cannot obtain a reliable filing timestamp, **lag every fundamental by a conservative fixed offset (e.g. 45 days after quarter end)** and say so in the write-up.

### Storage

| Store                | Technology                                     | Holds                                                                                           | Why                                                                                                                                                                 |
|----------------------|------------------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Operational / audit  | **PostgreSQL 16** (Docker)                     | Predictions, decisions, orders, trades, positions, experiment log, model registry, research log | Needs transactions, constraints and an append-only audit trail. This is your evidence; it must not be a file someone can edit.                                      |
| Research / analytics | **DuckDB over Parquet**                        | Bars, features, backtest outputs                                                                | Columnar scans over the full panel in milliseconds, zero-ops, and Parquet files are naturally immutable and hashable. The whole dataset fits comfortably in memory. |
| Raw snapshots        | **Dated directories + manifest**               | Instrument dumps, raw API responses, news HTML                                                  | Reproducibility. Every backtest cites a snapshot hash.                                                                                                              |
| Secrets              | OS keyring or `.env` with 600 perms; never git | API key/secret, TOTP, Telegram token                                                            | See §23.                                                                                                                                                            |

> **Deliberate non-choices**
>
> No Kafka, no Airflow, no Spark, no Feast, no vector database, no Kubernetes. 125,000 rows do not need distributed anything, and every one of these adds a failure mode between you and the answer. Use `systemd` timers and a Python function. If the project outgrows that, it will tell you.

### Universe definition and survivorship

Picking 50 names *today* and backtesting them over ten years is survivorship bias in its purest form: you have selected companies that stayed listed, stayed liquid, and stayed in the index. Handle it explicitly:

1.  **Define the universe by a rule, evaluated as of each date** — e.g. "F&O-eligible on date *T*, top 150 by 60-day median traded value, excluding the bottom decile by price". Store `universe_membership(as_of_date, symbol, rank, reason)`.
2.  **Run the backtest on the rule-based universe**, not on today's fixed 50. Then run it again on your chosen fixed 50 and report both. The gap between them is your survivorship premium — publish it.
3.  For the **live experiment**, a fixed 50 is fine: forward tests cannot have survivorship bias. Freeze the list, commit it, and do not change it mid-experiment (changing it is a form of look-ahead).
4.  **Sector diversification:** draw from at least 10 NIC/NSE sector groups with a cap of ~6 names per sector, so the panel has genuine cross-sectional variation rather than 50 correlated financials.

## 05 · Feature engineering plan

*You asked which features are actually useful rather than adding everything. Here is the honest triage, ordered by expected value per unit of implementation and leakage risk.*

> **Two transformations that matter more than any individual feature**
>
> **Cross-sectional normalisation.** On each day, convert every raw feature to its cross-sectional rank (or z-score) within the universe, then map ranks to \[-1, 1\]. This removes the market-wide level, makes features comparable across volatility regimes, and is what almost every successful cross-sectional model in the literature does. It is also the cheapest defence against non-stationarity.
>
> **Volatility scaling.** Divide return-based features by trailing realised volatility. A 2% move in a quiet stock and a 2% move in a volatile one are different events.

### Tier 1 — build first, expect most of the signal here

| Feature                                 | Definition                                                                               | Rationale & evidence                                                                                                                                             |
|-----------------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Short-horizon reversal**              | Cross-sectional rank of returns over 1, 2, 3, 5 days                                     | The dominant documented daily cross-sectional effect (Jegadeesh 1990). Krauss et al. use nothing else and get a working model.                                   |
| **Multi-horizon momentum**              | Cumulative returns over 10, 20, 60, 120, 240 days, ranked                                | Krauss et al.'s exact feature set. Momentum exists in India (IIM-A factor library) with crash risk.                                                              |
| **Idiosyncratic residual return**       | Residual from a rolling 60-day regression of stock return on NIFTY and its sector index  | Separates the predictable stock-specific component from the unpredictable market component — see §08. Arguably the single most valuable engineered feature here. |
| **Realised volatility & vol ratio**     | 5, 20, 60-day realised σ; ratio of 5d to 60d                                             | Conditions everything else; vol expansion often precedes reversal.                                                                                               |
| **Volume anomaly**                      | log(today's volume / 20-day median volume); traded-value rank                            | Volume spikes mark information arrival and are strongly interactive with reversal.                                                                               |
| **Overnight vs intraday decomposition** | Trailing sums of close→open and open→close returns, separately                           | Lou–Polk–Skouras. Cheap, and potentially the most informative structural feature in the set.                                                                     |
| **Position in range**                   | (close − 20d low) / (20d high − 20d low); distance to 52-week high                       | Well-documented anchoring effect; cheap.                                                                                                                         |
| **Beta and market-relative strength**   | 60-day beta to NIFTY; stock return minus beta × NIFTY return over 5/20 days              | Needed to interpret everything else, and required for risk attribution.                                                                                          |
| **Sector-relative return**              | Stock return minus its sector index return over 1/5/20 days                              | Isolates within-sector selection, which is the least market-driven part of your decision.                                                                        |
| **Correlation to NIFTY and to sector**  | Rolling 60-day correlation; and rolling average pairwise correlation across the universe | Average pairwise correlation is a good, cheap regime proxy — it spikes in stress.                                                                                |
| **Gap behaviour**                       | Open-to-previous-close gap, standardised by ATR; frequency of gap fills                  | Directly relevant if you trade close→open or open→close.                                                                                                         |
| **Illiquidity**                         | Amihud: mean(\|return\| / traded value) over 20 days                                     | Cost proxy and a risk control. Use it to *exclude*, not only to predict.                                                                                         |

### Tier 2 — market context, one row per day shared by all stocks

These do not vary cross-sectionally, so they cannot help you *rank* stocks — but they interact with stock features (a reversal signal behaves differently in a high-VIX regime) and they gate the "should we trade at all" decision.

- NIFTY 50 return (1/5/20d), realised vol, position vs 50/200-day MA
- **India VIX** level and percentile — the most useful single regime variable available to you
- Market breadth: fraction of universe up today; advance/decline over 5 days
- Average pairwise correlation across the universe (regime proxy)
- Sector index returns and sector dispersion (rotation proxy)
- Previous US session close (S&P 500, NASDAQ), **GIFT Nifty** level relative to NIFTY close
- USD/INR, Brent crude, gold, 10-year G-sec yield — daily changes
- IIM-A daily India factor returns (market, size, value, momentum)

> **On GIFT Nifty specifically**
>
> If you cut off features at 15:15 IST on day *T*, GIFT Nifty continues trading afterwards and overnight. Using its *15:15 value* is legitimate; using its next-morning value to predict the same morning's open is not. This is the single easiest place to leak in an Indian pipeline. Enforce it with the as-of contract, not with care.

### Tier 3 — fundamentals

**Blunt assessment: fundamentals are nearly worthless at a one-day horizon.** Valuation, ROE, leverage and earnings growth move on quarterly timescales; they cannot explain tomorrow's return. They matter here in exactly two ways:

- **As conditioning variables** — the reversal effect may differ between high- and low-quality names. Add a small number (P/E rank, P/B rank, ROE rank, debt/equity rank, market-cap rank) purely as interaction terms, and expect low importance.
- **As event flags** — "results due within 3 days", "results announced yesterday", "ex-dividend today", "in F&O ban period". **These are far more valuable than the ratios**, because they identify days when the return distribution is different, which is mostly a reason to *abstain*.

Do not build a fundamentals pipeline in v1. Build the corporate-events calendar instead.

### Tier 4 — news features

Covered in §06. Summary: valuable in principle, but you cannot honestly backtest them on historical scraped news, so they are a Phase-2 forward-validated experiment, not a v1 feature.

### How to decide what is actually useful

1.  **Univariate rank IC per feature**, computed on the training window only, with a Newey–West t-stat and Benjamini–Hochberg correction across features. Publish the table.
2.  **Feature stability**: recompute IC per year. A feature whose IC flips sign across sub-periods is noise, however good its full-sample number is.
3.  **SHAP importance across walk-forward folds.** Rank-correlate the importance vectors between folds; below ~0.5 means the model is fitting fold-specific noise.
4.  **Ablation, not addition.** Start from the Tier-1 set and remove groups; report ΔIC. Adding features one at a time to a growing set is how you overfit the selection process itself.
5.  **Hard cap the feature count.** With ~2,500 days and heavy cross-sectional correlation, your effective sample is far smaller than 125,000. **Target 25–40 features, not 300.**

## 06 · News-analysis pipeline

*Built correctly this is the most interesting part of the project. Built the usual way it is the most dangerous.*

> **Why historical news backtests are almost always invalid**
>
> 1.  **Timestamps are unreliable.** Aggregators restamp, edit and republish. An article "published 09:15" may have been updated at 14:30 with information from the day's trading.
> 2.  **Archives are survivorship-filtered.** Dead links, removed articles and paywalled history mean the past you can scrape is not the past that existed.
> 3.  **Coverage density changes over time**, so a "number of articles" feature partly encodes the calendar.
> 4.  **LLM scorers leak through their weights.** An LLM asked in 2026 whether a 2019 headline is bullish may simply know what happened next. No data-pipeline discipline can detect this.
>
> **Therefore:** start collecting news forward from day one, treat news models as a Phase-2 experiment validated only on forward-collected data, and do not let a news-based backtest number into any decision.

### Stages

| \#  | Stage                      | Implementation                                                                                                                                                                                                                                                                                                                                                                                                              | Output                                    |
|-----|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| 1   | **Collect**                | Poll every 15 min: NSE and BSE corporate announcements (the only genuinely authoritative, exchange-timestamped source), plus RSS from major Indian business publications and a broad aggregator feed. Store raw HTML/JSON with `first_seen_at` from your own clock.                                                                                                                                                         | `news_raw`                                |
| 2   | **Deduplicate**            | Canonicalise URL (strip UTM, resolve redirects) → SHA-256 exact match; then MinHash + LSH over 5-gram shingles of the headline+lede at ~0.85 Jaccard to cluster syndicated copies. Keep the earliest `first_seen_at` in each cluster as the cluster's timestamp.                                                                                                                                                            | `news_cluster`                            |
| 3   | **Timestamp**              | Record both the source's claimed `published_at` and your `first_seen_at`. **Use `min(first_seen_at, published_at + 15 min)` as `available_time`** — conservative on both sides. Flag any item whose claimed publish time precedes your first sighting by more than 24 h as untrustworthy.                                                                                                                                   | `available_time`                          |
| 4   | **Link to entities**       | Alias table per symbol (legal name, common name, ticker, brand names, subsidiary names) + fuzzy matching, then a relevance score: is the company the *subject* or a passing mention? An LLM classifier does this well and cheaply on a few hundred items a day.                                                                                                                                                             | `news_entity(news_id, symbol, relevance)` |
| 5   | **Classify events**        | Structured extraction rather than sentiment. Fixed taxonomy: *results, guidance, order win, M&A, management change, regulatory/legal, credit rating, analyst action, block/bulk deal, capex, dividend/buyback, product, macro-sector, other*. Plus polarity, surprise-vs-expectation where stated, and magnitude. Boudoukh et al. is the justification: identified events explain far more than undifferentiated sentiment. | `news_event`                              |
| 6   | **Score sentiment**        | FinBERT as a baseline, **validated on 200 hand-labelled Indian headlines first**. Add SESTM (Ke–Kelly–Xiu) once you have ≥6 months of forward-collected text, since it learns Indian-market-specific vocabulary from returns directly and cannot leak through pre-trained weights.                                                                                                                                          | `news_score(model_version, …)`            |
| 7   | **Aggregate to stock-day** | Within the window (previous cut-off, today's cut-off\]: count, relevance-weighted mean sentiment, max magnitude, event-type indicator flags, **novelty** (1 − max cosine similarity to that stock's last 30 days of news), **velocity** (count vs 30-day mean), **sentiment change** vs trailing mean, and sector-aggregate sentiment for the stock's peers.                                                                | Feature columns                           |

### Source reliability

Maintain a static `source_tier` weighting rather than trying to learn one: **Tier 1** exchange filings (authoritative timestamp, no spin); **Tier 2** established wire and business press; **Tier 3** aggregators and syndicators; **Tier 4** everything else, collected but excluded from features. Do not learn source reliability from returns until you have years of data — it is a fast route to fitting noise.

> **The realistic v1 news feature**
>
> If you build only one news feature, build **"has this stock had a Tier-1 exchange announcement in the last 24 hours, and of what type"**. It is exactly timestamped, free, unambiguous, unleakable, and it identifies the days when the return distribution genuinely differs. Everything else is a Phase-2 refinement of it.

------------------------------------------------------------------------

# Part V · Modelling and decision-making

*From a random baseline to a ranker, plus the target-formulation question you specifically wanted investigated, and the risk layer that stops a bad day becoming a bad month.*

## 07 · Model experimentation roadmap

*A ladder. You may not climb to rung *n+1* until rung *n* has been beaten out of sample, on the same folds, with the difference bootstrapped.*

| Rung | Model                                                                            | What it tests                                                                                                         | Effort | Expected outcome                                                                                              |
|------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------|
| 0    | **Uniform random pick**                                                          | The harness itself. Run it 1,000 times to get the null distribution of every metric you will later quote.             | hours  | Gives you the honest reference distribution. **Do this first.**                                               |
| 1    | **Always-long the index proxy** / equal-weight 50                                | Whether stock selection adds anything at all over just being invested.                                                | hours  | Surprisingly hard to beat net of costs.                                                                       |
| 2    | **Single-feature sorts** — 1-day reversal; 20-day momentum                       | Whether the classic effects exist in your universe and period at all.                                                 | 1 day  | If reversal has no IC here, stop and re-examine the data before modelling.                                    |
| 3    | **Ridge / logistic regression** on Tier-1 features, cross-sectionally normalised | Whether a linear combination beats single features. This is your real baseline.                                       | 2 days | Often within noise of the best model. Take that seriously.                                                    |
| 4    | **LightGBM regression** on demeaned forward returns                              | Whether non-linearity and interactions add anything.                                                                  | 2 days | The workhorse. Grinsztajn et al. says this class should win on tabular data.                                  |
| 5    | **LightGBM `lambdarank`** — daily query groups                                   | **Your ranking-vs-classification question, directly.** Same features, same folds, ranking objective.                  | 1 day  | Expect a modest but real improvement in Precision@K.                                                          |
| 6    | **CatBoost / XGBoost** as ensemble diversity                                     | Whether ensembling different tree implementations helps and whether their disagreement is a usable confidence signal. | 1 day  | Small gain in IC; the disagreement signal is the more interesting output.                                     |
| 7    | **Calibration layer** — isotonic or Platt on a purged validation fold            | Whether the probabilities mean anything. Required before any confidence-threshold rule.                               | 1 day  | Raw GBM probabilities are typically miscalibrated; this is cheap and materially improves the abstention rule. |
| 8    | **Ensemble** — rank-average of rungs 3–6                                         | Whether averaging beats the best single model out of sample.                                                          | hours  | Usually yes, modestly, and it is more stable across regimes.                                                  |
| 9    | **Sequence model** — DLinear first, then GRU/LSTM only if DLinear wins           | Whether path shape carries information beyond summary statistics.                                                     | 1 week | Low prior. Zeng et al. is the reason to try the linear version first.                                         |
| 10   | **News/text model** (Phase 2, forward data only)                                 | H4.                                                                                                                   | weeks  | Not evaluable before ~250 forward days.                                                                       |
| 11   | **Meta-model / abstention gate**                                                 | Whether a second-stage model can predict when the first stage is trustworthy.                                         | 1 week | Genuinely promising, but needs the base model to work first.                                                  |
| —    | ~~Transformers, graph networks, RL~~                                             | —                                                                                                                     | —      | **Explicitly out of scope** until §29 is passed. See §28.                                                     |

### Training protocol (identical at every rung)

- **Expanding window** with a minimum 3-year initial training set; retrain monthly, predict daily. Also test a 2-year rolling window — non-stationarity may favour forgetting.
- **Purged, embargoed cross-validation** for hyperparameters. With a 1-day label there is minimal label overlap, but a 1-day purge plus a 5-day embargo costs nothing and protects you when you later test multi-day holds.
- **Group by date, never shuffle.** Any CV split that puts the same date on both sides of the boundary leaks the market factor and will inflate your metrics enormously.
- **Fit every transformation inside the fold** — scalers, imputers, cross-sectional normalisation statistics, calibration. Fitting a scaler on the full sample is the most common leak in published stock-prediction papers.
- **Fixed seeds, logged.** Repeat each configuration with 5 seeds and report the spread — if seed variance exceeds the difference between two models, the difference is not real.

## 08 · Prediction formulation and ranking strategy

*You asked specifically whether ranking 50 stocks beats predicting each independently. Here is the argument, then the experiment that settles it for your data.*

### The decomposition that decides the answer

Write each stock's next-day return as

    r_i,t+1  =  α_i,t+1  +  β_i · r_mkt,t+1  +  γ_i · r_sector,t+1  +  ε_i,t+1

For a large-cap Indian equity, the market and sector terms typically account for **50–75% of daily return variance**. The market term is, to a first approximation, *unpredictable* — that is what four decades of failed market-timing research says.

Now consider what a binary "will stock *i* close green tomorrow?" classifier is being asked to do. Because `β_i · r_mkt` dominates, `P(r_i > 0)` is mostly `P(r_mkt > 0)`. A well-trained classifier will therefore spend most of its capacity trying to predict the market, fail, and settle on predicting roughly the unconditional base rate for every stock — which looks like ~53% accuracy and feels like a result, but contains no cross-sectional information whatsoever.

> **Conclusion**
>
> **Rank stocks on predicted cross-sectionally demeaned return.** Demeaning removes the common factor analytically, so the model only has to learn the part that is actually learnable. Ranking is then the natural objective because you only need the *ordering* to choose top-K, and ordering is a strictly easier target than either magnitude or sign.

### The four candidate formulations, compared

| Formulation                  | Target                                  | Pros                                                                                                                         | Cons                                                                                                                      | Verdict                                                                           |
|------------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Binary direction**         | `1[r > 0]`                              | Simple; gives a probability directly; matches your literal question                                                          | Dominated by the market factor; throws away magnitude; base rate varies daily so accuracy is uninterpretable              | **\[report only\]** Keep it as a reported diagnostic, not as the training target. |
| **Expected return**          | `r_i,t+1`                               | Preserves magnitude; needed for a cost threshold                                                                             | Heavy tails; a handful of extreme days dominate the loss; still contains the market factor                                | **\[with care\]** Winsorise at ±5σ and use Huber loss.                            |
| **Excess / demeaned return** | `r_i,t+1 − r̄_U,t+1`                     | Removes the unpredictable common factor; much better signal-to-noise; directly the quantity you can act on cross-sectionally | A positive excess return can still lose money in a falling market — you need a separate market gate for long-only trading | **\[primary training target\]**                                                   |
| **Rank / learning-to-rank**  | Position within the day's cross-section | Robust to outliers and to changing return scale; exactly matches "pick the top K"; optimises the metric you care about       | Loses magnitude, so a cost threshold needs a second calibration step mapping rank score → expected return                 | **\[primary decision layer\]**                                                    |

**Recommended architecture: train two heads on the same features.** A LightGBM ranker (`lambdarank`, one query group per date) produces the ordering, and a LightGBM regressor on demeaned returns produces a magnitude estimate calibrated to basis points. The ranker decides *which*; the regressor decides *whether the expected edge clears the cost threshold*. This resolves the tension cleanly and is about 40 extra lines of code.

### The experiment that answers H2

Same features, same folds, same seeds, three heads: (a) per-stock logistic classifier on raw direction, (b) regressor on demeaned return, (c) lambdarank. Compare on Precision@1, Precision@3, daily rank IC and NDCG@5, with a paired stationary bootstrap over days for the confidence interval. Report the table regardless of which wins — a negative result here is publishable within your own project and saves you from building the wrong thing.

### Selection rules and their trade-offs

| Rule                          | Trade-off                                                                                                                                                                          | Recommendation                                                                                                  |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Top-1**                     | Maximum concentration of predicted edge; maximum idiosyncratic variance. With a daily σ of ~1.8% and any realistic edge, single-name noise swamps the signal for hundreds of days. | **\[no\]** Only ~1 flat fee, but statistically the worst possible way to test a hypothesis.                     |
| **Top-3, equal weight**       | Cuts idiosyncratic variance by roughly √3 while giving up little predicted edge; 3× the ₹15.34 DP charge on sells.                                                                 | **\[recommended\]** The right balance for a ₹1.5–3 lakh experiment.                                             |
| **Top-5 to Top-10**           | Better statistics still, but the marginal names have much weaker predicted edge and per-scrip fees accumulate.                                                                     | **\[for evaluation\]** Run in the backtest to see how edge decays with K; too fee-heavy for small live capital. |
| **Confidence threshold**      | Trade only if calibrated probability exceeds a level. Reduces turnover; risks selecting only "easy" days that are also crowded.                                                    | **\[yes, in expected-return form\]**                                                                            |
| **Expected-return threshold** | Trade only if predicted edge \> round-trip cost + buffer. Directly economically motivated.                                                                                         | **\[yes — this is the abstention rule\]**                                                                       |
| **Risk-adjusted ranking**     | Rank by predicted return ÷ predicted volatility instead of raw predicted return. Tilts away from lottery names.                                                                    | **\[test as a variant\]** Cheap; often improves realised Sharpe more than it costs in mean.                     |
| **Sector constraint**         | At most one name per sector in the top-K. Costs a little predicted edge, buys real diversification.                                                                                | **\[yes\]**                                                                                                     |
| **Correlation constraint**    | Reject a candidate whose 60-day correlation with an already-selected name exceeds ~0.7 (Ledoit–Wolf shrunk estimate); or cluster the universe and take at most one per cluster.    | **\[yes, the clustering version\]** This is your "explicitly model correlation" requirement, done cheaply.      |
| **Market-regime filter**      | Skip trading when NIFTY is below its 200-day MA or India VIX is in its top decile. Intuitive; also a great way to overfit a handful of historical episodes.                        | **\[defer, and pre-specify\]** If you test it, fix the rule *before* looking, and test only 2–3 variants.       |

## 09 · Risk management and position sizing

*At this capital scale the dominant risks are operational and cost-related, not market-related. Size the controls accordingly.*

### Sizing

Use **inverse-volatility sizing with hard caps**, targeting equal risk contribution rather than equal rupees:

    target_notional_i = clamp(
        (risk_budget_per_position / sigma_20d_i) * price_i,
        min = max(50_000, 40 * dp_charge / cost_tolerance),   # fee-efficiency floor
        max = min(0.40 * equity, adv_cap_i)                   # concentration + liquidity caps
    )
    adv_cap_i = 0.001 * median_20d_traded_value_i             # never exceed 10 bps of ADV

The **fee-efficiency floor** deserves emphasis: the flat ₹15.34 DP charge is 0.31% of a ₹5,000 position and 0.015% of a ₹1,00,000 one. Below roughly ₹50,000 per position, depository fees alone consume any plausible edge. **This sets your minimum viable capital: about ₹1.5 lakh for a top-3 delivery strategy.** Below that, use intraday MIS (no DP charge) or do not trade live at all.

### Position-sizing schemes considered

| Scheme                                 | Assessment                                                                                               | Verdict                |
|----------------------------------------|----------------------------------------------------------------------------------------------------------|------------------------|
| Equal rupee weight                     | Simplest; over-weights risk in volatile names                                                            | **\[v1 fallback\]**    |
| Inverse-volatility (naive risk parity) | Equalises risk contribution; needs only a variance estimate, which is far more estimable than a mean     | **\[recommended\]**    |
| Full risk parity / HRP                 | Correct for many assets; overkill for K=3 and adds covariance estimation error                           | **\[defer\]**          |
| Mean-variance optimisation             | Extremely sensitive to expected-return estimates, which are precisely what you are least sure of         | **\[no\]**             |
| Full Kelly                             | Requires knowing your edge. You do not. Estimation error in the edge produces catastrophic over-betting. | **\[absolutely not\]** |
| Fractional Kelly (¼)                   | Defensible *after* you have a validated edge estimate with confidence intervals                          | **\[only post-§29\]**  |

### Exit rules

Your stated design is a fixed one-day hold, which is the cleanest thing to test because it makes the label unambiguous. Enrich it carefully:

- **Baseline: fixed hold, exit in the next CAS.** Label and trade coincide exactly. This is the scientifically correct default.
- **Overlap-aware rebalance** — if a held name is still in tomorrow's top-K, *do not sell and rebuy it*. This alone can cut turnover by 30–50% and is the single highest-value change to the strategy. It does mean the realised hold is variable, so keep the fixed-hold variant as the scientific reference.
- **Stop-loss:** a stop inside a one-day hold mostly harvests noise — intraday σ is comparable to the whole predicted edge. If you use one, make it wide (≥2× ATR) and treat it as a fat-tail guard, not a P&L tool. **Note that stop-loss orders are excluded from the closing auction**, so an unfilled stop simply carries no protection into the close.
- **Take-profit:** truncates the right tail of a distribution whose mean is what you are trying to measure. **\[omit in v1\]**

### Hard limits (enforced in code, not in judgement)

| Control                  | Limit                                                         | Action on breach                                                          |
|--------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------|
| Max notional per order   | 40% of equity                                                 | Reject order, alert                                                       |
| Max gross exposure       | 100% of equity (no leverage, no MTF)                          | Reject, alert                                                             |
| Max positions            | K + 1                                                         | Reject, alert                                                             |
| Max orders per day       | 2 × (K + 1)                                                   | Halt trading, alert                                                       |
| Symbol allowlist         | Frozen universe only                                          | Reject, alert loudly — this catches token-mapping bugs                    |
| Daily loss limit         | 2% of equity                                                  | Flatten at next window, halt for the day                                  |
| Peak-to-trough drawdown  | 8% of starting equity                                         | **Full stop.** Requires manual review and a written note before resuming. |
| Consecutive losing days  | 7                                                             | Halt, review — usually a data or execution bug, not bad luck              |
| Data staleness           | Any universe symbol with no tick in 10 min during 14:30–15:15 | Exclude symbol; if \>5 symbols, abstain for the day                       |
| Position/ledger mismatch | Any                                                           | Halt immediately. Never trade on top of an unreconciled book.             |
| Manual kill switch       | Presence of `HALT` file or Telegram command                   | No new orders; existing positions exit normally                           |

------------------------------------------------------------------------

# Part VI · Validation

*The backtester, the simulated broker, the metrics, and the baselines that any claimed edge must clear. This is where most projects like this quietly fail, usually without noticing.*

## 10 · Backtesting methodology

*Assume the backtest is lying until you have proved otherwise. The tests below are how you prove it.*

### Structure

A walk-forward loop over trading dates. For each date *T*: build features with `knowledge_cutoff = T 15:15`; if a retrain is scheduled, train only on data with `available_time < T`; call the *same* `strategy.decide` the live system calls; pass intents to the fill simulator; advance the portfolio; record everything. The output is not a return series — it is a full replay containing every feature vector, prediction and intent, so any later question can be answered without re-running.

### The fill simulator

| Aspect            | Modelling choice                                                                                                                                                                            |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Reference price   | For CAS-eligible names, the official close (auction equilibrium). Validate against NSE bhavcopy; do not assume Kite's candle close equals it.                                               |
| Slippage          | Base 5 bps per leg, scaled by `√(order_value / median_traded_value)` — the square-root impact law — with a floor of half a tick and a multiplier of 2× when India VIX is in its top decile. |
| Costs             | Full itemised Indian charge stack from a config file: STT, stamp, exchange transaction, SEBI, IPFT, GST, DP charge, brokerage. Never a single "0.1% cost" constant.                         |
| Fill probability  | Market orders fill. Limit orders fill only if the day's traded range crosses the limit, and then partially. Assume no fill for anything outside the ±3% auction band.                       |
| Capital           | Enforce available funds, integer share quantities, and T+1 settlement. Do not let the simulator buy with money it will not have.                                                            |
| Corporate actions | Apply ex-dates explicitly. A 3% ex-dividend drop is not a 3% loss; failing to handle this is a systematic negative bias in a long-only daily strategy.                                      |
| Halts & circuits  | If a symbol had no trade in the window or hit a circuit, the order does not fill. Log it as a missed trade — the fraction of missed trades is itself a result.                              |
| Missing data      | Never forward-fill a price into a feature without a staleness flag. If a symbol has no bar on day *T*, it is excluded from that day's cross-section, not imputed.                           |

> **The four tests that catch almost everything — build these before the model**
>
> 1.  **Shuffled-label placebo.** Randomly permute the labels within each day and re-run the entire pipeline. Mean IC must be indistinguishable from zero, and the strategy's Sharpe must sit inside the random-baseline distribution. If it does not, you have a leak. Run this in CI.
> 2.  **Synthetic-signal injection.** Add a known artificial predictor — e.g. `x_i,t = 0.3·r_i,t+1 + noise` — as a feature and confirm the pipeline recovers approximately the right IC and the right P&L magnitude. This validates that the harness can *detect* a signal, which a placebo test alone does not prove. Surprisingly rare, extremely valuable.
> 3.  **Time-shift test.** Shift every feature forward by one day (making it genuinely predictive by construction). The measured IC must jump dramatically. If it does not, your feature plumbing is broken.
> 4.  **Cost-sensitivity sweep.** Re-run with costs at 0×, 0.5×, 1×, 2× and 3× the modelled level. Plot net Sharpe against the multiplier. A strategy whose profitability disappears between 1× and 1.5× is not a strategy; it is a measurement of your cost assumptions.

### Validation design

- **Expanding-window walk-forward** across the full history, retraining monthly. This is the primary out-of-sample result.
- **Purged k-fold with embargo** for hyperparameter selection inside each training window only.
- **Combinatorial purged CV** if you want a distribution of backtest paths rather than one, which is a much better basis for confidence intervals — but it multiplies compute and the temptation to over-search. Optional.
- **A locked final holdout** of the most recent 24 months. Touch it exactly once, at the end, and record the date. If it disagrees with the walk-forward result, believe the holdout.
- **Sub-period stability.** Report IC and net Sharpe per calendar year and per volatility regime. A strategy that worked only in 2020–21 has told you something about 2020–21.

### The comparison set

Every strategy variant is reported against the same six references on the same dates with the same cost model — see §21 for the full list. The headline number is not "our Sharpe was X"; it is **"our net return minus the equal-weighted-50 net return was X, with a bootstrap 95% CI of \[a, b\], after a Deflated Sharpe correction for N configurations tried"**.

## 11 · Paper-trading / shadow methodology

*Zerodha has no sandbox, so you must build this. Its job is not to estimate profitability — the backtest does that — but to prove the *engineering* is sound while risking nothing.*

### What shadow mode is

The full live system runs on the real daily schedule against real live data and real broker endpoints for everything *except* order placement, which goes to a simulator implementing the same `Broker` protocol. Concretely, every day it:

- authenticates for real, fetches real instruments, real live quotes, real news;
- runs the real feature build at 15:14 and the real model at 15:15;
- records the intents it *would* have sent, with timestamps;
- records the actual market prices at 15:20, 15:25, the auction close, and next-day open — so implementation shortfall can be measured for orders that were never sent;
- marks the shadow portfolio to the real close and writes the same experiment-log row as live trading would.

> **The comparison that makes shadow mode worth the time**
>
> Each shadow day produces three numbers for the same decisions: what the **backtester** would have simulated, what **shadow mode** recorded live, and (later, in Stage 3) what **real fills** achieved. Systematic gaps between the first two are backtest bugs. Gaps between the second and third are execution reality. Both are far easier to diagnose here than with money on the line.

### Additional value: recording counterfactual execution

During shadow mode, log the price of every selected name at 15:16, 15:20, 15:25, the auction close, next-day 09:15, 09:20, 09:30, and next-day close. After 40 days you can answer empirically — with no risk taken — which execution window is cheapest and whether an intraday-confirmation entry beats a close entry. That data is worth more than any parameter you could tune.

### Exit criteria from shadow mode

- ≥40 trading days completed with zero unhandled exceptions.
- Zero days where the pipeline produced no decision for an avoidable reason.
- Reconciliation between shadow ledger and recomputed-from-prices ledger exact to the paisa on every day.
- Rank IC over the shadow period within the bootstrap confidence interval of the backtest OOS IC — i.e. live data does not look different from historical data.
- The backtester, re-run over exactly the shadow dates, reproduces the shadow returns to within a small tolerance.

## 20 · Evaluation metrics

*Split, deliberately, into the two questions from §00. Report both families every single day; never let one stand in for the other.*

### Family A — did the model predict?

| Metric                                | Definition                                                                                 | Why it is here                                                                                                              |
|---------------------------------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **Rank IC**                           | Daily Spearman correlation between predicted score and realised return, averaged over days | **The headline metric.** Scale-free, robust, directly measures cross-sectional ordering skill.                              |
| **ICIR**                              | Mean IC ÷ std of daily IC                                                                  | Consistency matters more than magnitude; an IC of 0.03 every day beats 0.10 one day in five.                                |
| **Newey–West t-stat on IC**           | With lag ≈ 5 to handle autocorrelation                                                     | The significance test. Bar is t \> 3.0 (Harvey–Liu–Zhu).                                                                    |
| **Precision@1, @3, @5**               | Fraction of top-K picks that were green — *and* that beat the day's cross-sectional mean   | Both versions. The second is the one that isolates skill.                                                                   |
| **Lift over daily base rate**         | Precision@K minus that day's fraction of green stocks                                      | **Raw accuracy is meaningless without this.** On a strong up day, random picking scores 90%.                                |
| **Pesaran–Timmermann statistic**      | Nonparametric test of directional predictive accuracy                                      | The correct significance test for a direction claim.                                                                        |
| **NDCG@5 / MRR**                      | Standard ranking metrics with return as gain                                               | Comparable to the learning-to-rank literature.                                                                              |
| **Brier score + reliability diagram** | Calibration of the probability output, decile-binned                                       | Required before any confidence-threshold rule can be trusted.                                                               |
| **Long–short decile spread**          | Mean return of top decile minus bottom decile, gross                                       | Measures the edge even though you can only trade the long side. **The purest measure of whether the model knows anything.** |
| **R²<sub>oos</sub>**                  | vs a zero-forecast benchmark, not vs the mean                                              | The Gu–Kelly–Xiu convention; expect a very small positive number if things are working.                                     |

### Family B — did the strategy make money?

| Metric                                                        | Note                                                                                                            |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Total return, CAGR, daily return distribution                 | Always net; report gross alongside to show the cost drag explicitly.                                            |
| Sharpe, Sortino                                               | Annualised from daily; state the risk-free assumption.                                                          |
| **Deflated Sharpe Ratio**                                     | **Mandatory.** Requires the honest trial count from `research_log`.                                             |
| Max drawdown, drawdown duration, volatility                   | Duration matters more than depth for whether you would actually have kept running it.                           |
| Win rate, average win / average loss, profit factor           | Watch for the classic pattern of high win rate with a fat left tail.                                            |
| **Alpha vs NIFTY**; alpha vs the IIM-A four-factor model      | The four-factor version is the honest one — it tells you whether you have alpha or repackaged momentum.         |
| Turnover; total costs as % of equity; cost/gross-profit ratio | **Report costs as a share of gross P&L.** If it exceeds ~50%, the design is wrong regardless of the net result. |
| Realised slippage vs modelled slippage                        | Per order, in bps. This validates the simulator.                                                                |
| Bootstrap CIs on everything                                   | Stationary bootstrap over days, ≥10,000 resamples. Point estimates alone are not a result.                      |

## 21 · Baselines

*Run all of these on identical dates, identical universe, identical cost model. A model that does not beat all of them has not earned its complexity.*

| Baseline                                     | Definition                                       | What beating it proves                                                                                                   |
|----------------------------------------------|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **B0 · Random**                              | Uniform random top-K each day, 1,000 simulations | Gives the null distribution of every metric. **Beating the mean is not enough — you must beat the 95th percentile.**     |
| **B1 · Buy-and-hold NIFTY**                  | Index proxy, no trading                          | That active management adds anything after costs                                                                         |
| **B2 · Equal-weight 50, rebalanced monthly** | The universe itself                              | That *selection* adds value beyond your universe-construction choices. This is the toughest and most important baseline. |
| **B3 · Always-long the highest-beta name**   | Trivial "predict market direction" proxy         | That you are not merely levering market exposure                                                                         |
| **B4 · Simple momentum**                     | Top-K by 20-day return                           | That the model beats the classic effect                                                                                  |
| **B5 · Simple mean-reversion**               | Bottom-K by 1-day return                         | **The one that matters at a daily horizon** — this is the effect your model is most likely to be silently rediscovering  |
| **B6 · Ridge on Tier-1 features**            | Rung 3 of §07                                    | That non-linearity earns its keep                                                                                        |

> **How to compare, properly**
>
> Do not compare Sharpe ratios by eyeballing. Use a **stationary bootstrap over days** to build the sampling distribution of the *difference* in net returns between your strategy and each baseline, and apply White's Reality Check / Hansen's SPA test across the whole family of variants you tried. Because you will have tried many models, the relevant question is never "is my best model good?" but "is my best model better than the best model I would expect to find by chance among this many attempts?"

## 13 · The 30-day live experiment

*Designed as a controlled engineering validation with a pre-registered protocol. Read the power analysis first — it changes what the experiment can honestly claim.*

> **Thirty days cannot establish a trading edge. Here is the arithmetic.**
>
> With a daily return standard deviation of ~1.8% on a concentrated position, the smallest mean daily return you could detect at t = 2 over *n* days is `2σ/√n`. Over 30 days that is **0.66% per day** — an annualised return in the hundreds of percent. Any 30-day P&L result, positive or negative, is noise.
>
> | What you want to detect               | Assumed true effect | Days needed for t = 2 |
> |---------------------------------------|---------------------|-----------------------|
> | Net daily return edge (σ = 1.8%)      | 0.05%/day           | ~5,200                |
> | Net daily return edge                 | 0.10%/day           | ~1,300                |
> | Net daily return edge                 | 0.20%/day           | ~325                  |
> | Daily rank IC (σ<sub>IC</sub> = 0.20) | IC = 0.02           | ~400                  |
> | Daily rank IC                         | IC = 0.03           | ~180                  |
> | Daily rank IC                         | IC = 0.05           | ~65                   |
> | Hit-rate lift, 50%→55% (80% power)    | 5 pp                | ~620 trades           |

**This is the strongest argument in the plan for making rank IC the primary metric.** It needs roughly an order of magnitude fewer observations than P&L to reach significance — and it is measured across all 50 stocks each day, not just the ones you traded. Note also that 50 stocks on one day are *not* 50 independent observations: with an average pairwise correlation of 0.3, the effective cross-sectional sample size is closer to 3. Correct for this when computing standard errors.

### What the 30 days is actually for

1.  **Engineering validation.** Does the system run unattended, every day, without intervention?
2.  **Execution calibration.** Is realised slippage inside the modelled band? Is implementation shortfall what the simulator predicted?
3.  **Distribution check.** Do live feature values, prediction scores and rank ICs look like the backtest's, or has something shifted?
4.  **Cost verification.** Do the actual contract-note charges match the cost model, to the rupee?
5.  **Discipline rehearsal.** Can you follow a pre-registered protocol for 30 days without changing the model mid-flight? Most people cannot. That is the real test.

### Protocol

| Rule            | Specification                                                                                                                                                                                                                                    |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model freeze    | Model artefact, feature set, config and universe are frozen and git-tagged before day 1. **No changes for 30 trading days.** Bug fixes to non-decision code are logged; anything touching `strategy.decide` ends the run and restarts the count. |
| Universe        | 50 names fixed and committed, ≥10 sectors, all F&O-eligible and liquid.                                                                                                                                                                          |
| Capital         | An amount whose total loss is irrelevant to you. Minimum ₹1.5 lakh for a top-3 delivery variant (§09); otherwise run the intraday variant or stay in shadow mode.                                                                                |
| Selection       | Top-3 equal-risk-weighted, with sector and correlation constraints, abstaining when predicted edge \< cost + buffer.                                                                                                                             |
| Execution       | Sell window 15:18, buy window 15:20 into CAS Session I, market orders with market protection, tagged with the decision ID.                                                                                                                       |
| Parallel shadow | **Shadow mode keeps running throughout**, unconstrained by capital or fills, so you always have an execution-free benchmark to difference against.                                                                                               |
| Analysis plan   | Written and committed *before* day 1: which metrics, which tests, which thresholds. No metric invented after seeing the data.                                                                                                                    |

### The daily experiment log

One row per stock per day (not one per trade), so the panel is analysable even on days you did not trade. Fields:

| Group          | Fields                                                                                                                                                                                                                                                        |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Identity**   | `run_id`, `trade_date`, `symbol`, `instrument_token`, `model_version`, `feature_set_version`, `config_hash`, `git_sha`, `data_snapshot_hash`                                                                                                                  |
| **Context**    | `nifty_close`, `nifty_ret`, `india_vix`, `regime_label`, `breadth`, `avg_pairwise_corr`, `sector`, `sector_index_ret`                                                                                                                                         |
| **Features**   | Full feature vector as JSONB, plus `features_asof_ts` and a per-symbol `staleness_flag`                                                                                                                                                                       |
| **Prediction** | `score`, `rank`, `prob_green`, `prob_beat_mean`, `expected_return_bps`, `pred_interval_lo/hi`, `ensemble_disagreement`                                                                                                                                        |
| **Decision**   | `selected`, `abstained`, `abstain_reason`, `gate_flags`, `target_qty`, `target_notional`, `sizing_basis`                                                                                                                                                      |
| **Execution**  | `order_id`, `broker_order_id`, `tag`, `entry_ts`, `entry_price`, `entry_qty`, `exit_ts`, `exit_price`, `decision_price` (the 15:15 reference), `slippage_bps`, `rejected_reason`                                                                              |
| **Outcome**    | `realised_return_c2c`, `realised_return_c2o`, `realised_return_o2c`, `excess_vs_universe`, `excess_vs_nifty`, `gross_pnl`, `brokerage`, `stt`, `stamp`, `exchange_txn`, `sebi_ipft`, `gst`, `dp_charge`, `net_pnl`, `direction_correct`, `beat_universe_mean` |
| **Provenance** | `news_count`, `news_available_time_min`, `corporate_event_flag`, `notes`                                                                                                                                                                                      |

Write this row for **all 50 symbols** every day. The untraded 47 are what give you the statistical power to answer the prediction question, and they cost nothing to record.

------------------------------------------------------------------------

# Part VII · Engineering specifics

*Schema, project layout, dependencies and external services. Concrete enough to start typing.*

## 14 · Database schema

*PostgreSQL for anything that constitutes evidence; Parquet/DuckDB for anything you scan in bulk. Abridged DDL — types and constraints shown where they encode a rule.*

    -- ============ reference & universe ============
    CREATE TABLE instrument (
      instrument_token  BIGINT PRIMARY KEY,
      tradingsymbol     TEXT NOT NULL,
      exchange          TEXT NOT NULL,
      segment           TEXT, name TEXT, tick_size NUMERIC, lot_size INT,
      first_seen        DATE NOT NULL, last_seen DATE NOT NULL,
      UNIQUE (tradingsymbol, exchange, first_seen)
    );

    CREATE TABLE universe_membership (
      as_of_date  DATE NOT NULL,
      symbol      TEXT NOT NULL,
      instrument_token BIGINT NOT NULL REFERENCES instrument,
      sector      TEXT NOT NULL,
      adv_20d     NUMERIC, rank_by_adv INT, fno_eligible BOOL NOT NULL,
      inclusion_reason TEXT NOT NULL,
      PRIMARY KEY (as_of_date, symbol)
    );

    -- ============ market data (mirrored to Parquet) ============
    CREATE TABLE bar_daily (
      instrument_token BIGINT NOT NULL, trade_date DATE NOT NULL,
      open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
      official_close NUMERIC,           -- from bhavcopy / CAS, may differ from `close`
      adj_factor NUMERIC DEFAULT 1.0,
      source TEXT NOT NULL, snapshot_id UUID NOT NULL REFERENCES data_snapshot,
      available_time TIMESTAMPTZ NOT NULL, ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (instrument_token, trade_date, snapshot_id)
    );

    CREATE TABLE corporate_action (
      instrument_token BIGINT NOT NULL, ex_date DATE NOT NULL,
      action_type TEXT NOT NULL,        -- split | bonus | dividend | rights | merger
      ratio NUMERIC, amount NUMERIC,
      detected_from TEXT,               -- 'snapshot_diff' | 'exchange_feed'
      available_time TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (instrument_token, ex_date, action_type)
    );

    CREATE TABLE market_context (
      trade_date DATE PRIMARY KEY, asof_ts TIMESTAMPTZ NOT NULL,
      nifty_close NUMERIC, nifty_ret NUMERIC, india_vix NUMERIC,
      usdinr NUMERIC, brent NUMERIC, gold NUMERIC, gsec10y NUMERIC,
      sp500_prev_close NUMERIC, giftnifty NUMERIC,
      ff_mkt NUMERIC, ff_smb NUMERIC, ff_hml NUMERIC, ff_wml NUMERIC,
      breadth NUMERIC, avg_pairwise_corr NUMERIC
    );

    -- ============ news ============
    CREATE TABLE news_raw (
      news_id BIGSERIAL PRIMARY KEY, url_hash TEXT UNIQUE NOT NULL, url TEXT,
      source TEXT NOT NULL, source_tier SMALLINT NOT NULL,
      headline TEXT NOT NULL, body TEXT,
      published_at TIMESTAMPTZ, first_seen_at TIMESTAMPTZ NOT NULL,
      available_time TIMESTAMPTZ NOT NULL,   -- min(first_seen, published + 15min)
      cluster_id BIGINT, timestamp_suspect BOOL DEFAULT FALSE,
      ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE news_entity (
      news_id BIGINT REFERENCES news_raw, symbol TEXT NOT NULL,
      relevance NUMERIC NOT NULL, is_subject BOOL,
      PRIMARY KEY (news_id, symbol)
    );
    CREATE TABLE news_score (
      news_id BIGINT REFERENCES news_raw, model_version TEXT NOT NULL,
      event_type TEXT, polarity NUMERIC, magnitude NUMERIC,
      novelty NUMERIC, scored_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (news_id, model_version)
    );

    -- ============ research provenance ============
    CREATE TABLE data_snapshot (
      snapshot_id UUID PRIMARY KEY, taken_at TIMESTAMPTZ NOT NULL,
      kind TEXT NOT NULL, content_hash TEXT NOT NULL, row_count BIGINT, notes TEXT
    );
    CREATE TABLE model_registry (
      model_version TEXT PRIMARY KEY, model_family TEXT NOT NULL,
      trained_at TIMESTAMPTZ NOT NULL, train_start DATE, train_end DATE,
      feature_set_version TEXT NOT NULL, hyperparams JSONB NOT NULL,
      git_sha TEXT NOT NULL, config_hash TEXT NOT NULL,
      data_snapshot_id UUID REFERENCES data_snapshot,
      cv_metrics JSONB, artifact_path TEXT NOT NULL, seed INT
    );
    CREATE TABLE research_log (          -- the multiple-testing counter
      trial_id BIGSERIAL PRIMARY KEY, logged_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      description TEXT NOT NULL, config JSONB NOT NULL,
      oos_ic NUMERIC, oos_sharpe NUMERIC, touched_holdout BOOL DEFAULT FALSE
    );

    -- ============ decisions & execution (the audit trail) ============
    CREATE TABLE prediction (
      run_id UUID NOT NULL, trade_date DATE NOT NULL, symbol TEXT NOT NULL,
      asof_ts TIMESTAMPTZ NOT NULL, model_version TEXT NOT NULL REFERENCES model_registry,
      score NUMERIC, rank INT, prob_green NUMERIC, prob_beat_mean NUMERIC,
      expected_return_bps NUMERIC, pi_lo NUMERIC, pi_hi NUMERIC,
      ensemble_disagreement NUMERIC, features JSONB NOT NULL,
      PRIMARY KEY (run_id, symbol)
    );
    CREATE TABLE decision (
      decision_id UUID PRIMARY KEY, run_id UUID NOT NULL, trade_date DATE NOT NULL,
      symbol TEXT NOT NULL, side TEXT CHECK (side IN ('BUY','SELL')),
      target_qty INT NOT NULL, target_notional NUMERIC,
      abstained BOOL NOT NULL DEFAULT FALSE, abstain_reason TEXT,
      gate_flags JSONB, sizing_basis JSONB, decision_price NUMERIC,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE order_record (
      order_id UUID PRIMARY KEY, decision_id UUID REFERENCES decision,
      broker_order_id TEXT UNIQUE, tag TEXT NOT NULL,
      variety TEXT, product TEXT, order_type TEXT, validity TEXT,
      qty INT, price NUMERIC, status TEXT NOT NULL, status_message TEXT,
      placed_at TIMESTAMPTZ, terminal_at TIMESTAMPTZ, raw_response JSONB
    );
    CREATE TABLE trade_record (          -- individual fills from /orders/:id/trades
      trade_id TEXT PRIMARY KEY, order_id UUID REFERENCES order_record,
      symbol TEXT, qty INT, price NUMERIC, filled_at TIMESTAMPTZ, raw JSONB
    );
    CREATE TABLE charge_record (         -- reconciled against the contract note
      trade_date DATE, symbol TEXT, brokerage NUMERIC, stt NUMERIC, stamp NUMERIC,
      exchange_txn NUMERIC, sebi_ipft NUMERIC, gst NUMERIC, dp_charge NUMERIC,
      total NUMERIC, source TEXT,        -- 'modelled' | 'contract_note'
      PRIMARY KEY (trade_date, symbol, source)
    );
    CREATE TABLE position_snapshot (
      snapshot_ts TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL,
      qty INT, t1_qty INT, avg_price NUMERIC, source TEXT,  -- 'ledger' | 'broker'
      PRIMARY KEY (snapshot_ts, symbol, source)
    );

    -- ============ the experiment log (§13) ============
    CREATE TABLE experiment_log (
      trade_date DATE NOT NULL, symbol TEXT NOT NULL, run_id UUID NOT NULL,
      stage TEXT NOT NULL CHECK (stage IN ('backtest','shadow','live')),
      -- identity, context, prediction, decision, execution, outcome columns per §13
      ...,
      PRIMARY KEY (trade_date, symbol, stage)
    );

> **Three schema rules worth defending**
>
> - **`available_time` is NOT NULL everywhere.** A row without it cannot be used in a feature, by construction.
> - **Predictions are written before decisions, and decisions before orders**, in separate transactions. If the process dies mid-day you can still reconstruct what the model said, which is the scientific record.
> - **`charge_record` stores both modelled and actual charges.** Reconciling them monthly is how you find out your cost model is wrong before your backtest does.

## 15 · Python project structure

*Flat where possible. The only hard rule: `kiteconnect` is imported in exactly one file.*

    greenclose/
    ├── pyproject.toml              # uv-managed, fully pinned via uv.lock
    ├── README.md
    ├── PREREGISTRATION.md          # committed and tagged before the live run
    ├── config/
    │   ├── base.yaml               # universe, model, risk params
    │   ├── charges.yaml            # every rate, dated, with a source URL
    │   ├── calendar.yaml           # NSE holidays, session times, CAS windows
    │   └── secrets.env.example
    ├── src/greenclose/
    │   ├── domain/                 # dataclasses only, no I/O
    │   │   ├── models.py           # Bar, NewsItem, Prediction, Intent, Fill, Position
    │   │   └── protocols.py        # Broker, DataSource, Clock  (typing.Protocol)
    │   ├── ingest/
    │   │   ├── instruments.py  bars.py  live.py  context.py  snapshots.py
    │   ├── news/
    │   │   ├── collect.py  dedupe.py  link.py  classify.py  aggregate.py
    │   ├── features/
    │   │   ├── registry.py         # name → (fn, deps, lookback, version)
    │   │   ├── price.py  market.py  news_feats.py  context.py
    │   │   └── build.py            # build(asof, store) -> DataFrame   [pure]
    │   ├── models/
    │   │   ├── datasets.py         # purged/embargoed splitters
    │   │   ├── train.py  infer.py  calibrate.py  registry.py
    │   ├── strategy/
    │   │   ├── decide.py           # THE shared decision function
    │   │   ├── selection.py  sizing.py  gates.py
    │   ├── execution/
    │   │   ├── kite_broker.py      # the ONLY file importing kiteconnect
    │   │   ├── sim_broker.py       # same Protocol, used by backtest + shadow
    │   │   ├── router.py  reconcile.py
    │   ├── backtest/
    │   │   ├── engine.py  fills.py  costs.py  corporate_actions.py
    │   ├── evaluation/
    │   │   ├── metrics.py  significance.py  attribution.py  report.py
    │   ├── ops/
    │   │   ├── alerts.py  health.py  killswitch.py  logging.py
    │   └── cli.py                  # typer: backfill | train | backtest | shadow | live | report
    ├── scripts/
    │   ├── daily_live.py  daily_shadow.py  weekly_retrain.py
    ├── notebooks/                  # exploration only; nothing here is ever imported
    └── tests/
        ├── test_pointintime.py     # cutoff enforcement
        ├── test_leakage.py         # shuffled-label, time-shift, synthetic-signal
        ├── test_costs.py           # against hand-computed Zerodha examples
        ├── test_fills.py           # simulator invariants
        ├── test_decide.py          # golden-file tests on frozen inputs
        └── test_reconcile.py

**Two conventions worth adopting.** First, `features/build.py` must be a pure function of (cut-off, store) with no network access — that makes it testable and makes backtest/live equivalence checkable. Second, `tests/test_decide.py` should be a golden-file test: freeze a day's feature matrix and portfolio, assert the exact intents produced. Any refactor that silently changes decisions then fails CI.

## 16 · Recommended libraries

| Purpose             | Choice                                                                                                   | Why this one                                                                                       |
|---------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Python / env        | **Python 3.12 + uv**                                                                                     | Fast, lockfile-based, reproducible. Reproducibility is a research requirement here, not a nicety.  |
| Broker API          | **pykiteconnect**                                                                                        | Official; includes the binary WebSocket parser you should not reimplement.                         |
| Dataframes          | **pandas** (+ **polars** optional)                                                                       | The panel is tiny. Use pandas for ecosystem compatibility; polars only if a specific step is slow. |
| Analytical store    | **DuckDB** + **pyarrow**                                                                                 | SQL over Parquet with zero infrastructure; ideal for walk-forward scans.                           |
| Operational store   | **PostgreSQL 16** + **SQLAlchemy** + **alembic**                                                         | Constraints and transactions for the audit trail; migrations you can review.                       |
| Models              | **LightGBM** (incl. `lambdarank`), **CatBoost**, **XGBoost**, **scikit-learn**                           | The evidence says gradient-boosted trees are the right default for tabular panels.                 |
| Deep learning       | **PyTorch** — only from rung 9                                                                           | Do not install it until you have a reason.                                                         |
| Statistics          | **statsmodels**, **scipy**, **arch**                                                                     | Newey–West standard errors, stationary bootstrap (`arch.bootstrap`), regime models.                |
| Interpretability    | **shap**                                                                                                 | TreeExplainer for the fold-stability diagnostic.                                                   |
| NLP                 | **transformers** + **FinBERT**, **datasketch** (MinHash), **trafilatura**, **feedparser**, **rapidfuzz** | Covers scoring, dedup, extraction, feeds and entity matching.                                      |
| Experiment tracking | **MLflow** (local file backend)                                                                          | Params, metrics and artefacts per run without hosting anything.                                    |
| CLI & config        | **typer**, **pydantic-settings**, **OmegaConf**                                                          | Typed, validated config — a config typo should be an error, not a silent behaviour change.         |
| Scheduling          | **systemd timers**; **APScheduler** in-process for intraday steps                                        | An orchestration platform would be more moving parts than the pipeline itself.                     |
| Logging & alerts    | **structlog** (JSON), **python-telegram-bot**                                                            | Structured logs are queryable; Telegram reaches your phone, which matters at 15:20.                |
| Testing             | **pytest**, **hypothesis**, **freezegun**                                                                | Property tests for feature invariants; frozen clocks for as-of tests.                              |
| Quality             | **ruff**, **mypy --strict** on `domain/` and `strategy/`                                                 | Type errors in the decision path are the expensive kind.                                           |

> **Libraries to think twice about**
>
> **backtrader / zipline / vectorbt** — general backtesting frameworks impose their own event model and none of them models Indian charges, T+1 settlement, DP fees or a closing auction. You will spend more time fighting the framework than writing the ~400 lines your own loop needs. **Qlib** — mine it for feature definitions, do not adopt it wholesale. **TA-Lib** — a C dependency for indicators you can write in five lines of pandas each, with full control over the as-of semantics. **Any "AI trading bot" repository** — the ones with stars are almost universally leaking.

## 17 · API and account requirements

| Requirement              | Detail                                                                                            | Cost              |
|--------------------------|---------------------------------------------------------------------------------------------------|-------------------|
| Zerodha trading account  | With demat, F&O activation not required for cash-only trading                                     | —                 |
| Kite Connect app         | Created at developers.kite.trade; gives api_key + api_secret; redirect URL must be registered     | ₹0 (orders)       |
| Market data subscription | Required for WebSocket + historical candles                                                       | ₹500/mo           |
| **Static IP**            | **Mandatory for order placement from 1 April 2026.** Up to 2 registered; changeable once per week | ₹300–800/mo (VPS) |
| Compute                  | 2 vCPU / 4 GB VM in an India region; that is ample for 50 stocks                                  | ₹800–1,500/mo     |
| Market context data      | Index and VIX via Kite; USD/INR, crude, gold, G-sec from a free provider; IIM-A factors free      | ₹0                |
| News                     | NSE/BSE announcements + RSS in v1; a paid news API only if Phase 2 justifies it                   | ₹0 → ?            |
| LLM API (optional)       | Event extraction and entity relevance on a few hundred items/day                                  | low               |
| Alerting                 | Telegram bot                                                                                      | ₹0                |

<span class="small">Running cost lands around **₹1,500–2,500 per month**. Note what that implies: on ₹1.5 lakh of capital, infrastructure alone is ~1.5% per year before a single trade. Include it in the net-return calculation — it is a real cost of the strategy, not an overhead you get to ignore.</span>

------------------------------------------------------------------------

# Part VIII · Operations

*How this fails, how it gets compromised, where it runs, and how you find out something is wrong before the market does.*

## 22 · Failure modes

*Ordered by how likely they are to silently invalidate the experiment rather than break it loudly. Silent failures are the dangerous ones.*

### Class 1 — silent scientific failures

| Failure                                           | How it shows up                                                   | Detection / mitigation                                                                 |
|---------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Look-ahead through the feature cut-off**        | Backtest Sharpe of 3+; live results nowhere near it               | Mandatory `knowledge_cutoff` argument; shuffled-label placebo in CI; time-shift test   |
| **Retroactive price adjustment**                  | Past backtest results drift when you re-download data             | Immutable dated snapshots with content hashes; alert on any change to a historical bar |
| **Scaler / calibrator fitted on the full sample** | Modest but consistent inflation of OOS metrics; very hard to spot | All transforms inside the fold; assert on split boundaries in tests                    |
| **Shuffled cross-validation on time series**      | Spectacular CV numbers, worthless model                           | Group-by-date splitters only; make shuffling impossible in the splitter API            |
| **Survivorship in the universe**                  | Backtest beats live persistently                                  | Rule-based point-in-time universe; report the fixed-50 vs rule-based gap               |
| **Fundamentals dated by period, not filing**      | Fundamental features show implausible predictive power            | Conservative 45-day lag; or omit fundamentals entirely in v1                           |
| **News timestamp drift / LLM weight leakage**     | News features look strong historically, vanish live               | Forward-collected news only for validation; `first_seen_at` from your own clock        |
| **Multiple-testing / backtest overfitting**       | The 30th variant you tried looks great                            | `research_log`, Deflated Sharpe, locked holdout, pre-registration                      |
| **Ignoring corporate actions on the holding day** | A systematic negative bias in a long-only daily strategy          | Explicit ex-date handling in the fill simulator and in realised-return computation     |
| **Base-rate confusion**                           | "68% accuracy!" on a month when the market rose                   | Always report lift over the daily cross-sectional base rate                            |

### Class 2 — operational failures

| Failure                                         | Mitigation                                                                                                                   |
|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Access token expired / login not done by 09:00  | Health check at 09:00 with a loud alert; system abstains rather than half-running                                            |
| Static IP changed (VPS rebuild, ISP change)     | Orders rejected. Assert the egress IP at startup against the registered value; remember changes are limited to once per week |
| WebSocket disconnect during the critical window | Auto-reconnect with backoff; REST quote fallback; staleness gate that abstains if \>5 symbols are stale                      |
| Duplicate orders from a naive retry             | Idempotency: unique `tag` per decision; check the order book for that tag before any retry                                   |
| Partial fills                                   | Reconcile from `/trades`, not from the order object; size subsequent legs from actual filled quantity                        |
| Margin / holding exceptions                     | Pre-trade funds and holdings check; treat as terminal, do not retry                                                          |
| Order outside the ±3% auction band              | Prefer market orders with market protection in Session I; verify status after 15:35                                          |
| Symbol suspended, in circuit, or in F&O ban     | Pre-trade tradeability check; fall through to the next-ranked candidate and log the substitution                             |
| BTST short delivery → auction penalty           | Log it as a distinct cost category; if it recurs, restrict to the most liquid quartile                                       |
| Clock skew or wrong timezone                    | `chrony`; `TZ=Asia/Kolkata`; store all timestamps as UTC with tz-aware types; assert market-hours sanity at startup          |
| Ledger/broker position mismatch                 | Immediate halt. Never trade on an unreconciled book.                                                                         |
| Disk full / DB down at 15:15                    | Health check at 14:30; if the store is unavailable, abstain — trading without a record is worse than not trading             |

### Class 3 — model and human failures

- **Concept drift** — feature distributions or the return-generating process shift. Monitor PSI on feature and score distributions; alert at PSI \> 0.2; investigate rather than auto-retrain.
- **Crowding** — short-term reversal is a widely known effect; any edge may decay precisely because others trade it. Track IC over rolling 60-day windows and expect decay.
- **Regime break** — a strategy fitted through a low-volatility period behaves differently in a shock. Report metrics conditioned on VIX tercile.
- **Mid-experiment tinkering** — the most likely failure of all. Changing the model on day 12 because of a bad week destroys the experiment. Enforce it with a git tag and a written rule, and treat any change as restarting the count.
- **Sunk-cost continuation** — deciding to keep trading because you built the system. Define the stop conditions in §29 *now*, while you are still objective.

## 23 · Security

*The threat model is narrow but the blast radius is your brokerage account.*

- **Credentials.** `api_key`, `api_secret`, TOTP seed and Telegram token live in the OS keyring or a 0600 `.env` outside the repo. A `pre-commit` secret scanner (gitleaks) is non-negotiable. Never log a token, never print a full response containing one.
- **Access token handling.** Held in memory and in a 0600 file with a `.gz`-free plain path; rotated daily by expiry anyway; wiped on shutdown. Treat it as equivalent to a session cookie for your trading account.
- **Do not automate the login credential flow.** The daily expiry is a deliberate regulatory control. Build a helper that opens the browser and captures the redirect; do not build something that stores your password and answers 2FA unattended. Beyond the terms question, it turns a compromised VM into a compromised brokerage account.
- **Defence in depth in code.** Hard-coded ceilings that no config can raise: max notional per order, max orders per day, symbol allowlist, product allowlist (CNC/MIS only), and an assertion that the total order value never exceeds a constant. These have caught real incidents in real systems; a config typo should cost nothing.
- **Host hardening.** SSH keys only, no password auth, firewall closed except SSH from your IP, unattended security upgrades, non-root service user, no inbound ports for the app itself.
- **Database.** Separate read-only role for analysis and notebooks; the trading service is the only writer to `order_record` and `trade_record`; nightly encrypted `pg_dump` to object storage.
- **Audit integrity.** Append-only intent: no `UPDATE` or `DELETE` on decision/order/trade tables from application code — corrections are new rows with a supersedes pointer. This is what makes the experiment credible later.
- **Data licensing.** Kite data must not be redistributed or displayed publicly — keep every dashboard private and behind auth.
- **Kill switch.** A `HALT` file on disk and a Telegram command, both checked before every order. Test it monthly.

## 24 · Deployment architecture

*One small VM. The static-IP rule makes this decision for you.*

    ┌─ India-region VM (2 vCPU / 4 GB), STATIC IP registered with Zerodha ─┐
    │                                                                      │
    │  systemd timers                    Docker                            │
    │   ├─ gc-login.timer      08:30      └─ postgres:16  (volume-backed)  │
    │   ├─ gc-premarket.timer  08:35                                       │
    │   ├─ gc-news.timer       */15min    Local filesystem                 │
    │   ├─ gc-live.timer       14:30       ├─ /data/parquet/   (bars,      │
    │   ├─ gc-eod.timer        16:30       │                    features)  │
    │   ├─ gc-report.timer     17:00       ├─ /data/snapshots/ (immutable) │
    │   └─ gc-retrain.timer    Sun 10:00   ├─ /data/mlruns/    (MLflow)    │
    │                                      └─ /data/models/    (artefacts) │
    │  Outbound: api.kite.trade, ws.kite.trade, news sources, Telegram     │
    │  Inbound:  SSH from your IP only                                     │
    └──────────────────────────────────────────────────────────────────────┘
            │ nightly encrypted pg_dump + parquet sync
            ▼
       Object storage (different provider/region)

- **Region:** an India region (Mumbai). Not for latency — EOD decisions do not need it — but for network reliability to the broker during the four-minute window that matters.
- **Static IP:** attach an elastic/floating IP *before* registering it, and never release it. Register the secondary IP too, so a host rebuild does not cost you a week of trading.
- **Deployment:** git pull + `uv sync` + systemd restart, via a single make target. No Kubernetes, no CI/CD pipeline for a single-node service.
- **Environments:** one VM, three modes selected by config — `backtest`, `shadow`, `live` — with the live mode requiring an explicit flag *and* the absence of the HALT file. Shadow and live run simultaneously during the 30-day experiment.
- **Backups:** nightly, tested quarterly by an actual restore. An experiment whose data you lost never happened.
- **Local development:** the same code runs on your laptop against the Parquet store for research; only the `live` mode requires the static IP.

## 25 · Logging and monitoring

### Logging

- **Structured JSON** via structlog, with `run_id`, `trade_date`, `stage` and `symbol` bound as context on every line, so a day can be reconstructed with one query.
- **Every external call** logged with request, redacted response, latency and retry count. When Zerodha and your records disagree, this is the tiebreaker.
- **Decision provenance**: log the model version, feature hash and gate evaluations for every decision, including abstentions. *Why you did not trade* is as much a result as why you did.
- Retention: 90 days of raw logs, indefinite for the database audit trail.

### Alerts (Telegram, three tiers)

| Tier             | Trigger                                                                                                                                                                   | Expectation      |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| **Critical**     | Position/ledger mismatch · order rejected · drawdown limit breached · daily loss limit breached · no access token by 09:00 · static IP mismatch                           | Act now          |
| **Warning**      | Data staleness · abstention triggered · PSI \> 0.2 on features or scores · realised slippage \> 2× modelled · unusual score distribution · news pipeline returned nothing | Look today       |
| **Daily digest** | 17:00: predictions, decisions, fills, net P&L, rank IC for the day, cumulative metrics, anything abstained and why                                                        | Read with coffee |

### Health checks

A heartbeat written at each pipeline stage, with a watchdog that alerts on a missed stage rather than waiting for you to notice. The most valuable single check: at 15:16, assert that *a decision row exists for today*. Silence is the failure mode you cannot see.

### The daily report

A single self-contained HTML file written to `/data/reports/YYYY-MM-DD.html` containing: the day's ranked predictions with scores; what was selected and why; fills with slippage against the decision price; itemised charges; realised returns for all three horizons; rank IC for the day and cumulatively with a confidence band; the score-distribution plot against its training distribution; and the running experiment scorecard against §29's criteria. Ten minutes a day reading this is what turns a system into a result.

------------------------------------------------------------------------

# Part IX · Building it, and knowing when to stop

*Sequence, stage gates, the ideas worth adding, and the conditions under which the honest answer is "no edge — write it up and move on".*

## 26 · Implementation roadmap

*Twelve weeks to a defensible answer, assuming evenings and weekends. Every phase ends with something you could show someone.*

| Wk    | Phase                            | Deliverable                                                                                                                                            | Done when                                                                                                      |
|-------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| 1     | **Data spine**                   | Kite auth, instruments, 10-year daily backfill for a rule-based universe + indices, snapshot store with hashing, Postgres schema, DuckDB/Parquet layer | You can query any stock's adjusted and unadjusted history as of any snapshot date                              |
| 2     | **Data validation**              | Bhavcopy reconciliation, corporate-action detection via snapshot diffs, holiday calendar, missing-data audit, close vs official-close study            | You can state, with numbers, how trustworthy your price data is                                                |
| 3     | **Backtest harness + baselines** | Walk-forward engine, fill simulator, full Indian cost model, B0–B5 baselines, metric library, bootstrap CIs                                            | Random and momentum baselines produce sensible, cost-inclusive numbers                                         |
| 4     | **Leakage harness**              | Shuffled-label, time-shift, synthetic-signal-injection and cost-sensitivity tests, wired into CI                                                       | All four tests pass on a deliberately-leaky variant *and* on the clean one, with the expected opposite results |
| 5     | **Tier-1 features**              | Price/market/context features with as-of enforcement; univariate IC table with multiple-testing correction; stability by year                          | You know which features have any standalone signal in *your* universe                                          |
| 6     | **Baseline models**              | Rungs 3–5: ridge, LightGBM regression, LightGBM lambdarank; the H2 ranking-vs-classification experiment                                                | You have an answer to "does ranking beat classification" with a confidence interval                            |
| 7     | **Horizon & cost study**         | C2C vs C2O vs O2C; hold periods of 1/3/5/10 days; overlap-aware rebalancing; the cost-multiplier sweep                                                 | You know which variant, if any, is net-positive — and by how much margin over the cost model                   |
| 8     | **Decision layer**               | `strategy.decide`, sizing, gates, sector/correlation constraints, abstention on cost threshold; golden-file tests                                      | Backtest runs entirely through the shared decision function                                                    |
| 9     | **Stage-1 verdict**              | Full walk-forward report, calibration, SHAP stability, DSR with honest trial count, locked-holdout evaluation                                          | **Gate 1 decision** — see §29                                                                                  |
| 10    | **Live plumbing**                | Kite broker adapter, WebSocket, order lifecycle, reconciliation, alerts, health checks, deployment on the static-IP VM                                 | A single one-share manual test order places, fills and reconciles end to end                                   |
| 11–12 | **Shadow mode**                  | Daily unattended shadow runs; counterfactual execution price logging; daily reports                                                                    | Rolls into the 40-day shadow requirement                                                                       |
| 13+   | **News pipeline (parallel)**     | Start collecting forward from week 1 regardless — the clock on news data starts the day you begin                                                      | ≥250 forward days before any news feature is evaluated                                                         |
| ~17   | **Gate 2 → 30-day live**         | Pre-registration committed, model frozen, small capital                                                                                                | **Gate 2 decision** — see §29                                                                                  |

> **Start the news collector in week 1, even though news is a Phase-2 feature**
>
> It is the only component whose value depends on wall-clock time rather than on your effort. Every week you delay is a week you cannot get back, and it costs almost nothing to run a 15-minute poller from day one.

## 27 · What to build first

*If you have one weekend, build these three things and nothing else.*

### 1 · The as-of data layer

Kite backfill into Parquet, with immutable dated snapshots and a `get_bars(symbols, cutoff)` accessor that *requires* a cutoff. Everything downstream inherits its correctness from this.

### 2 · The cost model

An exact, itemised Indian charge function, unit-tested against hand-computed Zerodha examples. It is 100 lines and it determines whether anything else is worth doing.

### 3 · The baseline backtest

Random, equal-weight, momentum and 1-day reversal, net of costs, with bootstrap confidence intervals. This tells you the shape of the problem before you have a model.

### 4 · The leakage harness

Shuffled-label and synthetic-signal tests. Build them *before* the model, not after — afterwards you will be motivated not to look too hard.

**The first genuinely informative result you can get** — probably in week 3 — is the net-of-cost performance of a 1-day reversal sort on your universe. If that classic, well-documented effect is negative after Indian costs, you have learned something important about the whole endeavour for two days of work.

## 28 · What NOT to build initially

*Each of these is defensible eventually and destructive now. The common thread: they add degrees of freedom before you have established there is anything to fit.*

- **Deep learning of any kind** — \[not now\] LSTM, GRU, transformer, graph network. Your effective sample size is far smaller than the row count suggests. Rung 9 exists for a reason and it is a long way up the ladder.
- **Reinforcement learning** — \[not now\] Needs a simulator at least as accurate as your backtester — which you have not yet validated — and it is the single easiest way to overfit a decade of daily data.
- **A historical news backtest** — \[never\] Not merely low-value: actively misleading, for the four reasons in §06. Collect news forward; do not scrape a fake past.
- **A fundamentals pipeline** — \[not now\] Weeks of work for features that cannot plausibly predict a one-day return. Build the corporate-events calendar instead, which is 5% of the work and most of the value.
- **Learned market-regime detection** — \[phase 2\] An HMM fitted on 2,500 days will find regimes; whether they generalise is another matter. If you want a regime gate, pre-specify a simple rule and test 2–3 variants, not 50.
- **Intraday data and intraday features** — \[phase 2\] Ten times the data volume and complexity for a decision made once a day. Add it only if the EOD version shows an edge that intraday confirmation might sharpen.
- **Options, futures, hedging overlays** — \[out of scope\] A NIFTY futures lot is far larger than the notional of this experiment, so hedging would dominate the position. Measure alpha statistically instead of hedging it out.
- **A web dashboard** — \[later\] The daily HTML report covers it. A dashboard is a project that consumes the time this project needs.
- **Dynamic universe selection** — \[between runs\] Changing the universe mid-experiment is a form of look-ahead and destroys comparability. Fix it; revisit between experiments.
- **Kelly sizing, leverage, MTF** — \[not now\] All three amplify an edge you have not yet demonstrated. Amplifying an unmeasured edge amplifies the estimation error, not the edge.
- **Hyperparameter mega-sweeps** — \[never\] Every configuration inflates the multiple-testing penalty. Use a small, pre-specified grid and log every trial.

## A · Ideas ledger — additions, triaged

*Your list plus mine, each with a verdict for version 1. Ordered by expected value, not by how interesting they are.*

### High value, cheap, do it in v1

- **Overlap-aware rebalancing (do not sell what you would rebuy)** — \[v1\] Cuts turnover 30–50% for ~20 lines of code. Given that costs are the binding constraint, this is the highest-return change available.
- **Cost-threshold abstention** — \[v1\] Trade only when predicted edge exceeds round-trip cost plus a buffer. Economically motivated, trivially implementable, and it directly implements your "decide whether to trade at all" idea.
- **Cross-sectional demeaning of the target** — \[v1\] Removes the unpredictable market factor from the label. Probably the single largest improvement in signal-to-noise you can make, and it is one line.
- **Synthetic-signal injection test** — \[v1\] Proves your harness can detect a signal of known size. Rarely done; catches whole classes of plumbing bugs that placebo tests miss.
- **Longer holding periods (3/5/10 days) as first-class variants** — \[v1\] The Blitz et al. prescription. Amortises cost across more days of expected return. May well be the difference between net-negative and net-positive.
- **Correlation-cluster constraint on top-K** — \[v1\] Cluster the universe on a Ledoit–Wolf-shrunk 60-day correlation matrix; at most one pick per cluster. Your "correlation-aware selection" requirement at 1% of the cost of a graph network.
- **Sector-relative and idiosyncratic-residual features** — \[v1\] Directly targets the learnable part of the return. Cheap and well-motivated.
- **Logging the full 50-stock panel every day, traded or not** — \[v1\] Multiplies your statistical power for the prediction question at zero marginal cost. The single best-value line in the experiment log.
- **Counterfactual execution-price logging** — \[v1\] Record prices at 15:16 / 15:20 / 15:25 / auction close / next open / +15 min. After 40 days you can answer "when should we execute?" empirically, with no risk taken.
- **Prediction calibration (isotonic / Platt on a purged fold)** — \[v1\] Required before any probability threshold means anything. Half a day of work.
- **Pre-registration + research_log + Deflated Sharpe** — \[v1\] The difference between a result and a story. Costs a text file and a database table.
- **Parallel shadow mode that never switches off** — \[v1\] Gives you a permanent execution-free benchmark to difference live results against, forever.
- **Volatility-scaled position sizing** — \[v1\] Equalises risk contribution using a variance estimate, which is far more estimable than a mean.
- **Corporate-events calendar (results dates, ex-dates, F&O ban)** — \[v1\] Mostly a reason to *abstain*. Cheap, exactly timestamped, and removes a category of unexplained losses.

### Worth doing, but after the base model works

- **Ensemble disagreement as a confidence signal** — \[phase 2\] Variance across model family / seed / bootstrap as an abstention input. Elegant and cheap once you have an ensemble — which is rung 6, not rung 1.
- **Split conformal prediction intervals** — \[phase 2\] Distribution-free uncertainty on expected return, ~20 lines. Better-founded than using raw model variance. Needs a calibration fold you will only have once training is stable.
- **Meta-model: predict when the base model is trustworthy** — \[phase 2\] Genuinely promising — the "should I trade today" question is often more learnable than "which stock". But it is a second model fitted to the first model's errors, so it needs a validated first model and its own OOS discipline.
- **Intraday confirmation before entry** — \[phase 2\] Enter next morning only if the stock has not already gapped past the predicted move. Plausible and cost-reducing, but requires intraday data and changes the label. Test on shadow-mode counterfactual data first — for free.
- **LLM event extraction from news** — \[phase 2\] Boudoukh et al. says structured events beat sentiment. The blocker is forward data, not capability. Start the collector now, evaluate in ~250 days.
- **SESTM (returns-supervised text scoring)** — \[phase 2\] Learns India-specific vocabulary from your own returns and cannot leak through pre-trained weights. Better-founded than FinBERT for this purpose, but needs a text corpus you do not yet have.
- **Drift detection (PSI on features and scores)** — \[monitor in v1, act in v2\] Monitor from day one, but *alert* rather than auto-retrain. Automatic retraining on drift is how a system quietly refits itself to a bad regime.
- **Regime-conditional models** — \[as features, not as models\] Splitting an already-small sample by regime multiplies overfitting risk. Prefer regime *features* and interactions inside one model over separate models per regime.
- **Long-short evaluation (top-K minus bottom-K)** — \[evaluate v1\] Even if you only trade long, the decile spread is the cleanest measure of whether the model knows anything. Evaluate it always; trade it only if intraday shorting proves viable.
- **SHAP feature-importance stability across folds** — \[phase 2\] A diagnostic, not a deliverable: unstable importance means you are fitting fold-specific noise. Cheap once the model exists.

### Defer or decline

- **Stock correlation graphs / GNNs** — \[decline\] The literature's own ablations suggest the ranking loss contributes more than the graph. On 50 stocks, a correlation matrix carries the same information.
- **Online / incremental learning** — \[decline\] Adds a moving target to an experiment whose point is a stable comparison. Weekly batch retraining achieves the same adaptivity with far better reproducibility.
- **Kelly criterion sizing** — \[post-§29\] Requires an edge estimate you do not have. Quarter-Kelly is defensible only after §29 passes with confidence intervals.
- **Full risk parity / HRP allocation** — \[defer\] Designed for many assets. With K=3, inverse-volatility weighting gets you 95% of the benefit with none of the covariance estimation error.
- **Dynamic universe selection** — \[between runs only\] Between experiments, yes. Within one, it is a look-ahead vector and it destroys comparability across days.

### Three additions of my own you did not list

- **Trade the residual, not the return** — \[v1\] Train on the residual from a rolling market+sector regression rather than on the raw or merely demeaned return. Demeaning removes the equal-weighted market; residualising removes each stock's *own beta exposure*, which is strictly better for a universe with dispersed betas. One extra regression per stock per day.
- **A "cost-aware label"** — \[phase 2, high value\] Instead of predicting `r`, predict `r − c` where `c` is that stock's modelled round-trip cost (which varies with liquidity and price). The model then naturally learns to prefer cheap-to-trade names, and the abstention threshold becomes "predicted value \> 0" rather than a hand-tuned constant.
- **Publish a negative result** — \[plan for it\] Write the result up properly whichever way it goes, with the pre-registration, the trial count and the cost sensitivity. A careful negative result on Indian daily cross-sectional predictability is genuinely more useful than most positive results in this literature, and it is the outcome you should be prepared for.

## 29 · Stage gates and success criteria

*Written now, while you are objective. Every threshold is checkable by a script, and the script should be written before the results exist.*

### Gate 1 · Backtest → shadow mode

| Criterion                                   | Threshold                                                |
|---------------------------------------------|----------------------------------------------------------|
| Walk-forward OOS length                     | ≥ 1,000 trading days                                     |
| Mean daily rank IC                          | \> 0.02, Newey–West t \> 3.0                             |
| IC sign consistency                         | Positive in ≥ 60% of calendar quarters                   |
| Precision@3 lift over daily base rate       | \> 0, with a bootstrap 95% CI excluding 0                |
| Net return vs equal-weight-50 (B2)          | Positive, bootstrap 95% CI excluding 0, after full costs |
| Beats B0's 95th percentile and all of B1–B6 | Yes, on the same dates                                   |
| Deflated Sharpe Ratio                       | \> 0 with the honest trial count from `research_log`     |
| Cost sensitivity                            | Still net-positive at 1.5× modelled costs                |
| Leakage suite                               | All four tests pass                                      |
| Locked holdout                              | Consistent with walk-forward (touched exactly once)      |
| Costs as share of gross P&L                 | \< 50%                                                   |

**If Gate 1 fails on net return but passes on IC** — the likeliest outcome — that is a real and reportable finding: *the model predicts, the strategy does not pay*. Response: extend the holding period, reduce turnover, retest. Do not respond by searching for a better model; that is how the multiple-testing count explodes.

### Gate 2 · Shadow → small live

| Criterion                                            | Threshold                                                         |
|------------------------------------------------------|-------------------------------------------------------------------|
| Shadow days completed                                | ≥ 40, ideally 60                                                  |
| Unhandled exceptions                                 | 0 in the final 20 consecutive days                                |
| Days with a decision produced                        | 100% (abstentions count as decisions)                             |
| Backtest replay vs shadow returns on identical dates | Agreement within 10 bps/day median absolute difference            |
| Shadow rank IC                                       | Within the backtest OOS IC bootstrap CI                           |
| Feature-distribution drift                           | PSI \< 0.2 on all Tier-1 features                                 |
| Reconciliation                                       | Exact on every day                                                |
| Modelled vs observed execution prices                | Median absolute gap \< 8 bps                                      |
| Capital available                                    | ≥ ₹1.5 lakh for delivery, and its total loss is irrelevant to you |
| Pre-registration                                     | Committed and git-tagged                                          |

### Gate 3 · Small live → anything larger

| Criterion                             | Threshold                                                                                         |
|---------------------------------------|---------------------------------------------------------------------------------------------------|
| Live days                             | ≥ 120 (not 30 — see the power analysis in §13)                                                    |
| Live rank IC                          | Statistically indistinguishable from backtest OOS IC                                              |
| Realised slippage                     | Within the modelled band; no systematic bias                                                      |
| Actual charges vs modelled charges    | Agree to within 5% on the monthly total                                                           |
| Net return vs B2 over the live period | Positive point estimate; CI need not exclude zero yet, but must not exclude the backtest estimate |
| Drawdown                              | Never breached the 8% limit                                                                       |
| Operational incidents                 | Zero unreconciled orders; zero limit breaches                                                     |

> **Stop conditions — define these now, honour them later**
>
> - Gate 1 fails after two honest attempts at cost reduction → **stop**, write up the negative result.
> - IC is positive but net return is negative at every holding period tested → **stop trading, keep predicting.** Continue logging predictions indefinitely; the dataset itself becomes the asset.
> - Live IC is significantly below shadow IC → **stop** and find the bug. Do not adjust the model.
> - You find yourself wanting to change the model mid-experiment → **stop the experiment** and restart the count. Wanting to change it is the signal.
> - Drawdown limit hit → **stop**, review in writing, and only resume with an explicit reason that is not "it should mean-revert".

### What "success" means, stated precisely

### Full success

Rank IC \> 0.02 with t \> 3 out of sample, positive net-of-cost spread versus equal-weight-50 with a CI excluding zero, DSR \> 0, and live results consistent with backtest. *Probability: low.*

### Partial success — the likely good outcome

Significant rank IC, calibrated probabilities, live matching shadow matching backtest — but net return not distinguishable from zero after costs. **This is a genuine scientific result and a well-built system.** Report it as such.

### Informative failure

No detectable IC, but the harness demonstrably works (placebo passes, synthetic signal recovered). You have established an upper bound on predictability in this universe with these features. Useful.

### Actual failure

You cannot tell whether there is an edge because the pipeline leaked, the costs were wrong, or the model changed mid-experiment. **Every design decision in this plan exists to prevent this outcome.**

## 30 · Future extensions

*Only if §29 passes. Listed in rough order of expected value.*

1.  **Horizon extension.** 3–10 day holds with banded rebalancing. The literature's strongest suggestion for making a short-horizon signal economically viable.
2.  **Universe expansion** to the full F&O-eligible set (~200 names) with a point-in-time liquidity rule. More cross-sectional information is the cheapest way to increase statistical power — 200 stocks × 250 days beats 50 × 1000 for cross-sectional estimation.
3.  **Intraday shorting** to build a genuine cross-sectional long-short, which isolates alpha from market beta and roughly doubles the information per day.
4.  **Forward-validated news and event features**, once ≥250 days of collected data exist.
5.  **Execution optimisation** using the counterfactual price log — auction vs open vs staged entry, informed by measured shortfall rather than intuition.
6.  **Cost-aware training objective** and turnover-penalised portfolio construction.
7.  **Multi-horizon ensemble** — combine 1-day, 5-day and 20-day models, weighted by their respective net Sharpe.
8.  **Meta-model / abstention model** once there is a validated base model to abstain from.
9.  **Regime-conditional feature interactions** (as features, not as separate models).
10. **Sequence and graph models** — only after the tabular ladder is exhausted and only with the expanded universe.
11. **Reinforcement learning for execution** (not for selection) — a much better-posed RL problem than portfolio choice, and one where the simulator can actually be validated.

## ✓ · Closing note

The most valuable thing this project can produce is not a profitable strategy — it is a **trustworthy measurement**. Almost every retail attempt at this fails not because the market is unpredictable but because the measurement apparatus is broken: leakage, unrealistic costs, base-rate confusion, and quiet mid-experiment tinkering. Get the apparatus right and you will get an honest answer, which is worth far more than a backtest that looks good.

Two sentences to keep pinned above the desk: *the model predicting stocks that go green and the strategy making money are different claims, and both need separate evidence.* And: *if you cannot say how many configurations you tried, you do not have a result.*

<span class="small">Facts about Kite Connect, Zerodha charges, the Closing Auction Session and SEBI's retail algo framework were verified against primary sources on 30 August 2026 and are cited inline throughout. All of them change — re-verify before the live stage, and keep every rate in `config/charges.yaml` with a date and a source URL.</span>
