# Implementation plan: NYC Airbnb price prediction

*A complete, phase-by-phase build plan implementing the full MLOps pipeline (Git → DVC → FastAPI → Docker → MLflow → Prefect → GitHub Actions) on a real regression problem. Written to be handed to Claude Code and executed phase by phase.*

---

## 1. Project overview

**Problem:** predict a listing's nightly price (in USD) from its location, room type, and booking-activity features.

**Type:** regression (the series' first — every prior project was binary classification, so RMSE/MAE/R² replace accuracy/precision/recall throughout).

**Dataset:** New York City Airbnb Open Data (2019), sourced from Inside Airbnb, distributed on Kaggle as `AB_NYC_2019.csv`. Confirmed: **48,895 rows, 16 columns**.

**Why this project:** real, freely available data; a universally intuitive question ("what should I charge for my listing"); a genuine change of shape from every prior project in the series (regression instead of classification); and enough real messiness (missing values, extreme outliers, a skewed target) to be a legitimate step up in difficulty, not just a reskin.

---

## 2. Dataset details

**Exact columns**, as confirmed directly:
```
id, name, host_id, host_name, neighbourhood_group, neighbourhood,
latitude, longitude, room_type, price, minimum_nights, number_of_reviews,
last_review, reviews_per_month, calculated_host_listings_count, availability_365
```

**Known real quirks** (confirm each by inspecting the actual downloaded file before writing cleaning code — don't assume these numbers are exact for whatever copy gets downloaded):
- `name`, `host_name`, `last_review`, `reviews_per_month` all contain missing values.
- `reviews_per_month` is missing exactly when `number_of_reviews == 0` (a listing with no reviews has no review rate) — fill with `0`, not the mean.
- `price` contains `$0` values (almost certainly data errors, not real free listings) and extreme outliers up to `$10,000`. The distribution is strongly right-skewed.
- `id`, `host_id` are identifiers, not predictive features. `name`, `host_name` are free-text/PII-adjacent and not used as model inputs.

**Target column:** `price`. Given the skew, train on `log1p(price)` and invert with `expm1()` when reporting/serving predictions — note this explicitly in `train.py` so it isn't a silent surprise later.

**Feature set for modeling:**
| Type | Columns |
|---|---|
| Numeric | `latitude`, `longitude`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365` |
| Categorical | `neighbourhood_group`, `neighbourhood`, `room_type` |
| Dropped | `id`, `name`, `host_id`, `host_name`, `last_review` |

`neighbourhood_group` has 5 values (Manhattan, Brooklyn, Queens, Bronx, Staten Island). `room_type` has 3 (Entire home/apt, Private room, Shared room). `neighbourhood` has over 200 values — this is the categorical-cardinality problem the series hasn't hit yet; use `OneHotEncoder(handle_unknown="ignore")` as usual, and note in the article/writeup that high-cardinality categoricals are a real, common wrinkle worth calling out.

---

## 3. Project structure

```
airbnb_price_prediction/
├── .gitignore / .dvcignore
├── docker-compose.yml              (Phase 10)
├── Dockerfile
├── requirements.txt
├── .github/workflows/
│   ├── ci.yml
│   └── deploy.yml
├── data/
│   └── AB_NYC_2019.csv (+ .dvc)
├── train.py                        (single baseline run)
├── track_experiments.py            (multi-run MLflow tracking)
├── orchestrate_training.py         (Prefect flow)
├── scripts/
│   ├── ci_seed_model.py
│   └── trigger_deploy.py
├── schemas.py
├── main.py
├── tests/
│   ├── test_schemas.py
│   └── test_model_registry.py
└── models/
    └── model.pkl
```

---

## 4. Phase-by-phase build plan

### Phase 1 — Git setup
- `git init`, `.gitignore` (standard Python + `data/`, `models/`, `mlruns/`, `mlartifacts/`, `mlflow.db`).
- `requirements.txt`: pin `scikit-learn`, `pandas`, `numpy`, `joblib`, `fastapi`, `uvicorn`, `pydantic`, `mlflow`, `prefect`, `pytest`, `requests` to whatever versions are actually installed and tested — don't guess pins, verify by running `pip show`.
- Commit in small, real steps as each phase completes, not one giant commit at the end.

### Phase 2 — DVC
- `dvc init`, then `dvc add data/AB_NYC_2019.csv` **before** any blanket `git add -A` (the Article 7.5 sequencing lesson — check this order explicitly).
- Configure a local remote, `dvc push`.

### Phase 3 — Baseline training script (`train.py`)
- Load data, clean per Section 2 (fill `reviews_per_month`, drop `$0` and top-outlier prices — decide and document a concrete threshold, e.g., drop `price == 0` and cap or drop above the 99th percentile), drop unused columns.
- `ColumnTransformer`: `StandardScaler` on numeric, `OneHotEncoder(handle_unknown="ignore")` on categorical.
- Baseline model: `LinearRegression`, wrapped with the preprocessor in one `Pipeline`.
- Train on `log1p(price)`; report RMSE/MAE/R² on the original price scale (invert predictions with `expm1` before scoring) — this inversion step is easy to get backwards, so write a quick sanity check comparing a hand-computed value.
- Save with `joblib` to confirm the pipeline works end to end before anything else touches it.

### Phase 4 — Pydantic schema and FastAPI app
`schemas.py` — one `Listing` input model, fields matching the feature set exactly (not the raw CSV columns — no `id`/`name`/etc.):
```python
class Listing(BaseModel):
    neighbourhood_group: Literal["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    neighbourhood: str  # too many values for Literal — validate against a known-list check instead if desired
    latitude: float
    longitude: float
    room_type: Literal["Entire home/apt", "Private room", "Shared room"]
    minimum_nights: int = Field(..., ge=1)
    number_of_reviews: int = Field(..., ge=0)
    reviews_per_month: float = Field(..., ge=0)
    calculated_host_listings_count: int = Field(..., ge=1)
    availability_365: int = Field(..., ge=0, le=365)

class PricePrediction(BaseModel):
    predicted_price: float
    currency: str = "USD"
```
`main.py` — same shape as the series' `main.py`: load the model once at startup via `MLFLOW_TRACKING_URI` + `models:/AirbnbPriceModel@champion` (not a hardcoded path — Article 12's lesson), `/health`, `/predict` returning `PricePrediction`. Remember the `expm1` inversion happens here too if the model was trained on log-price.

### Phase 5 — Dockerfile
- Reuse the series' proven pattern exactly: `python:3.11-slim`, dependencies copied and installed before code (layer caching), train during build or copy a pre-trained model in — decide which and document why.

### Phase 6 — MLflow experiment tracking (`track_experiments.py`)
Run at least these configurations, logging params/metrics/model for each, all under one experiment name (`airbnb-price-prediction`):
| Run name | Model | Key params |
|---|---|---|
| `linreg_baseline` | LinearRegression | defaults |
| `rf_100` | RandomForestRegressor | `n_estimators=100` |
| `rf_300_depth10` | RandomForestRegressor | `n_estimators=300, max_depth=10` |
| `gb_100_lr01` | GradientBoostingRegressor | `n_estimators=100, learning_rate=0.1` |
| `gb_200_lr005` | GradientBoostingRegressor | `n_estimators=200, learning_rate=0.05` |

Log `rmse`, `mae`, `r2` for every run (computed on the original price scale, per Phase 3's inversion note). Point `MLFLOW_TRACKING_URI` at a real running server, not the default local store — confirm this actually works by checking the run appears at the server's URL before moving on.

### Phase 7 — MLflow Model Registry
- Register the best run (lowest RMSE, but sanity-check against MAE/R² too — with regression it's worth explicitly deciding and documenting which metric is the deciding one, since they won't always agree) as `AirbnbPriceModel`.
- Promote it to the `champion` alias. Use aliases, not the deprecated stage API.

### Phase 8 — Prefect orchestration (`orchestrate_training.py`)
- One flow: `load_data` → `split_data` → `train_and_log` (×5 configs) → `promote_best_model`.
- `@task(retries=2, retry_delay_seconds=5)` on `load_data`.
- `MLFLOW_TRACKING_URI` read from environment, not hardcoded.
- Schedule via `.serve(cron=...)` once the manual run is confirmed working.

### Phase 9 — Tests
- `test_schemas.py`: valid listing accepted; an invalid `room_type` rejected; `minimum_nights=0` rejected; a negative `latitude`/`longitude` — decide whether to bound these (NYC's real lat/long range is roughly 40.5–40.9 / -74.25–-73.7; consider `Field` bounds reflecting that, since a listing claiming to be in Antarctica shouldn't validate).
- `test_model_registry.py`: load the model via the real registry URI (not `joblib`), predict on one clearly-cheap and one clearly-expensive synthetic listing, assert the predicted price direction makes sense (e.g., a Manhattan entire-home listing should predict higher than a Bronx shared room, all else equal) — this is a good regression-specific sanity check, since there's no clean "0.5 threshold" the way classification had.

### Phase 10 — GitHub Actions CI/CD
- `ci.yml`: on pull_request → checkout, setup-python (with `cache: pip`), install deps, start an ephemeral MLflow server, run `scripts/ci_seed_model.py` (trains one quick baseline and registers/promotes it so `AirbnbPriceModel@champion` resolves), run `pytest`, then a `build-and-push` job gated by `needs: test`.
- `deploy.yml`: `workflow_dispatch`-only, triggered by `scripts/trigger_deploy.py` from the Prefect flow after promotion — mirror Article 12's pattern exactly, including using OIDC if pushing to ECR, or repository secrets for Docker Hub.

### Phase 11 (optional, advanced) — Unify with Docker Compose
- Same architecture as Article 12: `mlflow-server` and the API as two Compose services on one network, `MLFLOW_TRACKING_URI=http://mlflow-server:5000` as the one env var every component reads, a healthcheck-gated `depends_on` so the API doesn't race the MLflow server's ~30-second startup.

---

## 5. Definition of done

Each phase is complete when its own claim is actually verified, not assumed:
- Phase 3: the saved model's RMSE/MAE/R² are printed and sane (a plausible RMSE is on the order of tens of dollars, not thousands — if it's wildly off, the log-price inversion is the first thing to check).
- Phase 6: all 5 runs appear in the MLflow UI with correct params and metrics.
- Phase 7: `mlflow.sklearn.load_model("models:/AirbnbPriceModel@champion")` from a **separate process** returns working predictions.
- Phase 9: running `pytest` shows all tests passing, and deliberately breaking one schema constraint produces a real, visible test failure (don't just trust that a test *would* fail — break something and watch it fail once).
- Phase 10: opening a real PR against the repo shows the CI workflow actually running and reporting a status check.
