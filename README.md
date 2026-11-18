# Bitcoin Volatility Forecasting: HAR vs. GARCH vs. LightGBM

> Does machine learning actually beat classical volatility models out of sample? And if it does, what is it really learning?

I built a full Databricks pipeline that turns **3.5 million 1-minute Bitcoin bars** into daily realized volatility, then pits **10 next-day forecasting models** against each other over **1,003 out-of-sample days**, using walk-forward validation and Diebold-Mariano significance tests.

**Short answer:** LightGBM wins on paper, but a plain linear model given the same features ties it exactly. The edge came from better inputs, not a better algorithm.

---

## TL;DR

| # | Finding | Evidence |
|---|---|---|
| 1 | **Intraday data beats daily data.** HAR-RV (288 five-minute returns per day) beats GARCH and GJR-GARCH (one return per day) by ~25% in QLIKE. | p < 0.001 |
| 2 | **No meaningful leverage effect in BTC.** GJR-GARCH is statistically identical to plain GARCH. | p = 0.73 |
| 3 | **HAR's blind spot is the calendar.** One weekend dummy cuts HAR's loss by 18% and recovers ~85% of LightGBM's gain. | p < 0.001 |
| 4 | **The "ML gain" is a feature gain.** LightGBM (full) beats standard HAR by 22%, but linear HAR-X with the same features matches it. LightGBM using only HAR's inputs is 15% *worse* than linear HAR. | p = 0.90 (LightGBM vs. HAR-X) |

---

## Results

Out-of-sample QLIKE loss over 1,003 days (lower is better):

| Rank | Model | QLIKE | vs. HAR rolling |
|---|---|---|---|
| 1 | **HAR-X** (linear, extra features) | **0.2107** | -22.0% |
| 2 | LightGBM (full features) | 0.2118 | -21.6% |
| 3 | HAR + weekend dummy | 0.2205 | -18.3% |
| 4 | HAR rolling (730d), the benchmark | 0.2700 | n/a |
| 5 | HAR expanding | 0.2707 | +0.3% |
| 6 | HAR (smearing correction) | 0.2723 | +0.9% |
| 7 | LightGBM (HAR features only) | 0.3114 | +15.3% |
| 8 | GJR-GARCH | 0.3587 | +32.9% |
| 9 | GARCH(1,1) | 0.3608 | +33.6% |
| 10 | Naive (tomorrow = today) | 0.4399 | +62.9% |

### Is the difference real? Diebold-Mariano tests

A negative mean difference means Model A has the lower loss. HAC standard errors are used throughout.

| Model A | Model B | Mean diff | t-stat | p-value | Takeaway |
|---|---|---|---|---|---|
| HAR rolling | HAR expanding | -0.0007 | -0.47 | 0.63 | Tie |
| GJR-GARCH | GARCH | -0.0022 | -0.35 | 0.73 | Tie |
| HAR rolling | GJR-GARCH | -0.0887 | -5.77 | < 0.001 | HAR wins |
| GARCH | Naive | -0.0791 | -3.15 | 0.002 | GARCH wins |
| HAR rolling | Naive | -0.1699 | -9.58 | < 0.001 | HAR wins |
| LightGBM (HAR feat.) | HAR rolling | +0.0414 | +5.42 | < 0.001 | HAR wins |
| LightGBM (full) | HAR rolling | -0.0582 | -4.04 | < 0.001 | LightGBM wins |
| HAR + weekend | HAR rolling | -0.0495 | -4.66 | < 0.001 | Weekend dummy helps |
| HAR-X | HAR + weekend | -0.0098 | -3.02 | 0.003 | Extra features help |
| LightGBM (full) | HAR-X | +0.0011 | +0.13 | 0.90 | Tie |

Every claimed finding survives a Bonferroni correction for 10 tests (threshold 0.005).

![Bitcoin realized volatility history](images/realized_vol.png)

---

## What each model is

| Model | Idea |
|---|---|
| **Naive** | Tomorrow's realized variance equals today's. |
| **HAR (expanding / rolling)** | Regress next-day log RV on daily, 7-day, and 30-day average RV ([Corsi, 2009](#references)). Rolling uses a 730-day window. Refit every day. |
| **GARCH(1,1) / GJR-GARCH** | Fit on daily percent returns, Student-t errors, zero mean, 730-day rolling window. Refit every day. |
| **LightGBM (HAR features)** | Gradient boosting on HAR's three inputs only. A fair test of "is the algorithm better?" |
| **LightGBM (full)** | HAR inputs plus downside semivariance share, daily return, absolute return, volume ratio, and next-day day of week. Refit weekly. |
| **HAR + weekend** | Linear HAR plus a Saturday/Sunday dummy. |
| **HAR-X** | Linear HAR plus every extra feature LightGBM gets. A fair test of "is the *information* better?" |

**Why 7 and 30 days instead of Corsi's 5 and 22?** Those numbers are a trading week and month for equities. Bitcoin trades every day, so a calendar week and month are 7 and 30.

---

## Why these results make sense

**HAR beats GARCH because it sees more.** GARCH only learns about today from one close-to-close return. On 2020-03-13, BTC closed up about 15%, yet that day has the highest realized volatility in the whole sample because of a violent intraday crash and rebound. A daily return can't see that. Realized variance can.

**The weekend effect is the big missing piece.** HAR has no idea what day tomorrow is, so it overpredicts quiet weekends and underpredicts Mondays. Next-day day of week was LightGBM's top feature by gain. A possible explanation is that the effect strengthened after U.S. spot Bitcoin ETFs launched in January 2024, tying more trading to equity market hours, but I did not test this.

**Trees lose to OLS here.** The HAR relationship is nearly linear, and with only ~730 training rows, trees approximate it with coarse steps. Once the linear model gets the same features, it captures everything LightGBM does.

**Rolling vs. expanding HAR tie** because HAR forecasts from recent volatility *levels*, so it follows the long downward drift in BTC volatility since 2021 automatically.

**Full-sample HAR coefficients** (interpretation only, R² = 0.555):

| Term | Coefficient | p-value |
|---|---|---|
| Constant | -0.787 | < 0.001 |
| Daily | 0.412 | < 0.001 |
| Weekly | 0.389 | < 0.001 |
| Monthly | 0.110 | 0.001 |

The coefficients sum to 0.91, so volatility is very persistent but mean-reverting. The monthly term is small, which suggests BTC volatility shocks fade faster than in typical equity HAR studies.

---

## Method

- **Realized variance:** the sum of squared 5-minute log returns per UTC day (288 per day). Five minutes balances microstructure noise against sample size. Sanity check: mean RV (0.00114) is within 11% of mean squared daily return (0.00103), so RV and GARCH target the same quantity.
- **Target:** next-day log RV, which tames RV's heavy right skew. Forecasts are converted back to variance with a bias correction (lognormal adjustment for baseline HAR, Duan smearing elsewhere).
- **Walk-forward validation:** for each origin day from 2024-01-01 to 2026-09-29, every model trains only on data available at that day's close and forecasts the next day. That gives 1,003 forecasts per model. No random splits, and no tuning on the test period.
- **Loss functions:** QLIKE (primary) and MSE, both robust to a noisy variance proxy ([Patton, 2011](#references)).
- **Significance:** Diebold-Mariano tests on daily QLIKE differences with HAC standard errors.

---

## Data

| | |
|---|---|
| Source | [Binance public data](https://github.com/binance/binance-public-data), BTCUSDT spot, 1-minute klines |
| Period | 2020-01-01 to 2026-09-30 (2,465 days) |
| Rows | 3,547,275 one-minute bars (99.93% of all minutes) |
| Files | 81 monthly archives, downloaded programmatically |

### Data quality issues I had to handle

- **Timestamp units changed mid-dataset.** Starting 2025-01-01, Binance spot timestamps switched from milliseconds (13 digits) to microseconds (16 digits). Parsed naively, 2025+ data lands tens of thousands of years in the future. Bronze keeps raw values; Silver normalizes to milliseconds. Verified: exactly 918,720 microsecond rows = 638 days x 1,440 minutes.
- **Exchange outages.** 2,325 minutes are missing, 2,310 of them in 15 distinct outages (longest: about 6 hours on 2020-02-19). Fourteen fell in 2020-2021, one on 2023-03-24, none since.
- **Handling.** Returns spanning an outage are flagged (`is_gap`) rather than deleted, and daily rows carry `has_gap`. None of the 10 highest-volatility days contain a gap, so outages don't create fake spikes.

---

## Pipeline

```
Binance archive (81 monthly zips)
        |
        v
[ Unity Catalog Volume ]   raw zips + extracted CSVs
        |
        v
[ Bronze ]  bronze_btc_1m      raw 1-min bars, explicit schema, lineage columns
        |
        v
[ Silver ]  silver_btc_1m      normalized timestamps, UTC, deduplicated
            silver_btc_5m      5-min closes, log returns, gap flags
        |
        v
[ Gold ]    gold_btc_daily_rv  daily realized variance, returns, volume, gap flag
        |
        v
[ Models ]  forecasts          every model's forecasts in one Delta table
        |
        v
[ Evaluation ]  QLIKE, MSE, Diebold-Mariano
```

**Engineering choices**
- **Idempotent ingestion:** downloads and extraction skip existing files, so any step can be re-run safely.
- **Explicit schemas:** Bronze never relies on inference, and Delta schema enforcement blocks accidental changes.
- **UTC session time zone:** daily boundaries follow the crypto convention. Without it, every daily value would silently shift.
- **Single forecasts table:** each model writes with `replaceWhere`, so models never overwrite each other and re-runs never duplicate rows. Rows are keyed by `target_date` (the day being forecast) to avoid off-by-one errors.
- **Right tool for the size:** PySpark for ingestion and 5-minute aggregation, pandas/statsmodels for the 2,465-row daily table.
- **No GARCH feature in LightGBM:** GARCH forecasts only exist from 2024 onward, leaving no history to train on.

---

## Limitations

- One asset (BTC) and one exchange (Binance spot).
- One test window (2024-2026); other regimes may behave differently.
- LightGBM refits weekly while HAR and GARCH refit daily, a small handicap for LightGBM.
- LightGBM hyperparameters were fixed and conservative, not tuned.
- The ETF explanation for the weekend effect is a hypothesis, not a result.
- Volume comes from a single exchange and may not reflect total market activity.
- All models overpredict mean RV by 6-12%. QLIKE punishes underprediction more, so a slight upward lean is expected.

## Ideas for next steps

- Tune LightGBM on a pre-2024 validation window.
- Test whether the weekend effect actually strengthened after the January 2024 ETF launch.
- Repeat on ETH or other assets and on additional exchanges.
- Explain the volatility spikes visible in 2024 and 2025 and check how each model handled them.

---

## Repository structure

```
notebooks/
  00_download_raw     Download and extract 81 monthly Binance archives into a UC Volume
  01_bronze_ingest    Load CSVs into Bronze with explicit schema and lineage columns
  02_silver           Normalize timestamps, deduplicate, build 5-min returns and gap flags
  03_gold             Aggregate to daily realized variance; plot volatility history
  04_baseline         Naive and HAR-RV walk-forward forecasts; create forecasts table
  05_garch            GARCH(1,1) and GJR-GARCH walk-forward forecasts
  06_evaluation       Leaderboard and Diebold-Mariano tests
  07_lightgbm         LightGBM, HAR + weekend, HAR-X, feature importance
images/
  realized_vol.png
README.md
```

## Reproducing

Built on Databricks Free Edition (serverless compute, Unity Catalog).

1. Create the schema `workspace.vol_project` and a managed volume `raw` inside it.
2. Run notebooks `00` through `07` in order. Each installs its own dependencies in its first cell.
3. Run `06_evaluation` last to rebuild the leaderboard and significance tests.

**Stack:** PySpark, Delta Lake, Unity Catalog, Databricks, pandas, NumPy, statsmodels, arch, LightGBM, matplotlib

## References

- Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review*.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
- Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. *Journal of the American Statistical Association*.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- Patton, A. J., & Sheppard, K. (2015). Good volatility, bad volatility: Signed jumps and the persistence of volatility. *Review of Economics and Statistics*.
