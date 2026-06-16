# Rossmann Store Sales Forecasting

> Single LightGBM model, 11.5% average error (RMSPE), top 5% of the [Kaggle leaderboard](https://www.kaggle.com/c/rossmann-store-sales/leaderboard) (3,738 teams).

Forecasting daily sales for ~1,115 retail stores six weeks ahead, using the public
[Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset
(German drugstore chain, 2013-2015).

This dataset mirrors the structural forecasting challenge faced by grocery
retailers: a fleet of stores with distinct sizes and assortments, recurring
weekly promotions, school/state holidays, and nearby competitors, all of which
move sales. Accurate store-level forecasts drive staffing, inventory, and
promotional-spend decisions.

---

## 1. Business problem

> Given two and a half years of daily sales history, predict each store's sales
> for the next six weeks.

Errors are scored with RMSPE (Root Mean Squared Percentage Error), the
official Kaggle metric. RMSPE is scale-invariant: a 10% miss on a small store
counts the same as a 10% miss on a large one, which is exactly the right incentive for a
chain where every store must be stocked correctly regardless of size.

## 2. Approach

| Stage | Choice | Why |
|-------|--------|-----|
| Validation | 3-fold walk-forward CV (expanding window across 2015) | No shuffling, that would leak the future. Each validation block is 6 weeks, matching the Kaggle test horizon. |
| Target | `log1p(Sales)` | Training MSE on the log target approximates the scale-invariant RMSPE. |
| Features | Calendar, holiday-proximity, promotion (incl. date-aware Promo2), competition, sales lags/rolling, and per-store aggregates | Section 4 |
| Primary model | LightGBM with a 30-trial Optuna sweep on Fold 1 | Fast on ~1M rows, native categorical handling, strong on tabular retail data. |
| Comparison model | MSTL (Nixtla `statsforecast`) at store-type cluster level | Interpretable trend + weekly + yearly decomposition. Prophet was deprecated by Meta in 2024. `statsforecast` is the maintained, faster successor. |
| Baselines | Naive (last week) and per-(store, weekday) median | The median is the bar a competent analyst sets without ML. The model must beat it convincingly. |

### Anti-leakage discipline (the part that actually matters)

- Lags are computed before closed-day filtering. Closed Sundays create
  calendar gaps. If you filter first, `sales_lag_7` silently points to the
  previous open day instead of 7 calendar days back. A dedicated test
  (`tests/test_features.py::test_lag7_respects_calendar_across_closed_days`)
  pins this.
- Per-store aggregates are fitted on the training fold only, then applied to
  validation/test (`StoreAggregates.fit/transform`).
- `Customers` is never used raw. It exists in `train.csv` but not
  `test.csv`. Its footfall signal is carried to inference via the
  `StoreDowAvgCustomers` aggregate (mean customers per store x weekday, learned
  on train).

## 3. Results

Run `make train` (LightGBM) and `make baseline` to populate `outputs/metrics_report.md`.
RMSPE is reported as mean +/- std across the 3 folds.

RMSPE measures average percentage error per store per day. Lower is better.

| Model | Avg error | vs naive |
|-------|-----------|----------|
| Naive (last week) | 34.6% +/- 2.0% | baseline |
| Median per (store, weekday) | 24.7% +/- 1.2% | 29% better |
| MSTL (cluster-level) | 66.3% +/- 21.1% | 91% worse |
| LightGBM | 11.5% +/- 2.1% | 67% better |

LightGBM fold breakdown:

| Fold | Period | Avg error |
|------|--------|-----------|
| 1 | Jan-Mar 2015 (Q1 slump) | 14.5% |
| 2 | Apr-Jun 2015 | 9.8% |
| 3 | Jun-Jul 2015 | 10.2% |

Fold 1 is harder: Q1 captures the January post-Christmas slump with limited
lag history from the prior year. Folds 2 and 3 are more typical conditions.

MSTL is included as a decomposition tool, not a store-level forecaster. Fitting
one trend curve across 200-300 heterogeneous stores per cluster cannot capture
individual store variation, which is why its error exceeds even the naive model.
The value is in its interpretable seasonal components (see notebook 03), not
in its point forecasts.

Tuned hyperparameters (Optuna, 30 trials on Fold 1):
`num_leaves=398, min_child_samples=20, feature_fraction=0.61, bagging_fraction=0.61`

## 4. Feature engineering

- Calendar: day-of-week and month capture the dominant weekly/seasonal
  rhythm. Day-of-month and month-boundary flags proxy the pay-day spending spike.
  `DaysToChristmas` is a non-linear ramp capped at 60 days.
- Holidays: `StateHoliday` encoded. `DaysSinceHoliday` / `DaysToPublicHoliday`
  capture pre- and post-holiday shopping. A per-Easter flag was deliberately
  dropped (only 3 Easters in the data, effectively noise) and folded into the generic
  holiday-proximity feature.
- Promotions: the standard `Promo` flag, plus a correctly decoded
  `Promo2Active`. A store's rolling Promo2 is only active when the month is in
  its `PromoInterval` and the date is past its start week, which is a common mistake to
  get wrong. Also includes a promotion-fatigue count and a promo x weekday interaction.
- Competition: `log1p(distance)` (the effect is logarithmic, not linear),
  months since a competitor opened (cap is data-driven), and a has-competitor
  flag.
- History: sales lags at 7/14/28/365 days and trailing rolling mean/std (all
  `shift(1)` to avoid same-day leakage).
- Store aggregates: per-store mean/median sales, per-(store, weekday) mean,
  promotional-sensitivity ratio, and the `Customers` proxy described above.

## 5. Project structure

```
src/rossmann/        importable, tested library code
  config.py          paths, column groups, hyperparameters
  data/              loader, cleaner, walk-forward splitter
  features/          calendar, promotions, competition, lags + StoreAggregates
  models/            base, baseline, lgbm (Optuna), statsforecast (MSTL)
  evaluation/        rmspe / mae / mape + LightGBM custom eval
  pipeline.py        load -> features -> filter -> CV orchestration
scripts/             train.py / evaluate.py / predict.py  (thin CLIs)
notebooks/           01_eda, 02_feature_analysis (SHAP), 03_results
tests/               metrics, features (incl. temporal-lag), splitter, pipeline
```

The transformation logic lives in `src/` and is unit-tested. Notebooks and
scripts only call it. No feature engineering is hidden inside a notebook cell.

## 6. Reproducing results

```bash
# 1. Environment (creates .venv, installs everything)
make install
#    macOS only: LightGBM needs the OpenMP runtime
brew install libomp

# 2. Data: download the three CSVs into data/raw/
#    from https://www.kaggle.com/c/rossmann-store-sales/data
#    (train.csv, test.csv, store.csv), or with the Kaggle CLI:
#    kaggle competitions download -c rossmann-store-sales -p data/raw && unzip -o 'data/raw/*.zip' -d data/raw

# 3. Run
make test       # full unit-test suite
make baseline   # median baseline RMSPE
make train      # LightGBM + Optuna, writes outputs/metrics_report.md
make predict    # writes outputs/submission.csv (Kaggle format)
```

A pinned lockfile is intentionally not committed. Generate one with
`pip freeze > requirements.txt` if you need byte-for-byte reproducibility.

## 7. What I'd do next

- Probabilistic forecasts (e.g. a Temporal Fusion Transformer or quantile
  LightGBM) to drive safety-stock decisions rather than point estimates.
- Loyalty-card / transaction features to replace the `Customers` proxy with
  real forward-looking footfall signal.
- Hierarchical reconciliation so store-level forecasts sum coherently to
  distribution-center demand.
