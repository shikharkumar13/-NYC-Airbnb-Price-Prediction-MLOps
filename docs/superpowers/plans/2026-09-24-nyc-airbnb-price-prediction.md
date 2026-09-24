# NYC Airbnb Price Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a beginner-friendly, end-to-end MLOps regression project (Git → DVC → training → FastAPI → MLflow tracking + registry → Docker → Prefect → GitHub Actions → Compose) that predicts the nightly price of a NYC Airbnb listing.

**Architecture:** A single shared module (`features.py`) holds every data-cleaning and model-building rule so that the baseline script, the MLflow experiment runner, the Prefect flow, and the CI seeding script can never drift apart. Models are trained on `log1p(price)` via `TransformedTargetRegressor`, so `.predict()` always returns dollars — the API and tests never hand-roll `expm1`. The FastAPI service does not bake a model into its image; it loads `models:/AirbnbPriceModel@champion` from whatever `MLFLOW_TRACKING_URI` points at, so promoting a new champion needs a restart, not a rebuild.

**Tech Stack:** Python 3.11 (uv-managed venv, matches `python:3.11-slim` image), pandas, scikit-learn, MLflow 3.x (server + registry with aliases), FastAPI + Pydantic v2, Prefect 3, DVC, pytest, Docker / Docker Compose, GitHub Actions, Docker Hub.

**Source spec:** `Implementation_Plan_NYC_Airbnb_Price_Prediction.md` (repo root). Section/phase numbers below refer to it.

## Global Constraints

- Dataset: `data/AB_NYC_2019.csv`, 48,895 rows × 16 columns (verified 2026-09-24 against `~/Downloads/Airbnb NYC 2019.csv`).
- Target: `price`. Train on `log1p(price)`, report and serve in USD via `expm1`.
- Numeric features: `latitude, longitude, minimum_nights, number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365`.
- Categorical features: `neighbourhood_group, neighbourhood, room_type` → `OneHotEncoder(handle_unknown="ignore")`.
- Dropped: `id, name, host_id, host_name, last_review`.
- `reviews_per_month` NaN → `0` (never the mean).
- Cleaning thresholds: drop `price == 0`; drop `price > 800` (≈ the 99th percentile, $799).
- MLflow experiment name: `airbnb-price-prediction`. Registered model: `AirbnbPriceModel`. Alias: `champion` (aliases only — never the stage API).
- `MLFLOW_TRACKING_URI` is always read from the environment; never hardcoded in Python.
- Metrics `rmse`, `mae`, `r2` computed on the original dollar scale.
- Docker base image: `python:3.11-slim`; dependencies copied and installed before code.
- Pin dependency versions to what is actually installed (read from `uv pip freeze`), never guessed.
- Commit after each task — small, real commits.
- Run everything from the repo root with the venv active (`source .venv/bin/activate`).

## Deviations from the source spec (and why)

| Spec says | This plan does | Why |
|---|---|---|
| `.gitignore` includes `data/` | Do **not** ignore `data/`; DVC writes `data/.gitignore` itself | Ignoring `data/` blocks `git add data/AB_NYC_2019.csv.dvc`, silently breaking DVC tracking. |
| Invert with `expm1` manually in `train.py` and `main.py` | `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` inside the model | The inversion travels *with* the model into the registry, so no consumer can forget it (train/serve skew). A hand-computed test still proves it. |
| Flat scripts only | Adds `features.py` (shared data/pipeline code) and `registry.py` (MLflow naming + promotion) | DRY: four entry points need the same cleaning and promotion logic. |
| Phase 5 (Docker) before Phase 6/7 (MLflow) | Docker task runs **after** the registry task | The image loads `@champion` at startup; it can't be verified until a champion exists. |
| Phase 9 tests at the end | Tests are written with each component (TDD) | Each task ends with its own passing tests. |
| CI trains on the DVC-tracked CSV | CI trains on a committed 2,000-row, PII-free sample `tests/fixtures/listings_sample.csv` | The DVC remote is local to this laptop; GitHub runners can't `dvc pull` it. |
| CI `build-and-push` job | CI **builds** the image (no push); `deploy.yml` pushes | Pushing unreviewed PR images to a public registry is poor practice; the Dockerfile still gets verified on every PR. |
| One `requirements.txt` | `requirements.txt` (dev/CI) + `requirements-serve.txt` (image) | Keeps Prefect/pytest out of the serving image; uses `mlflow-skinny` there. |

## File Structure

```
NYC-Airbnb-Price-Prediction/
├── .gitignore  .dockerignore  .dvcignore   pytest.ini   README.md
├── requirements.txt              # dev + CI (pinned)
├── requirements-serve.txt        # API image only (pinned, same versions)
├── Dockerfile
├── docker-compose.yml            # Task 13 (optional)
├── .github/workflows/ci.yml      # PR: test + build image
├── .github/workflows/deploy.yml  # workflow_dispatch: build + push to Docker Hub
├── data/AB_NYC_2019.csv(.dvc)    # DVC-tracked
├── features.py                   # constants, load/clean/split, build_model, evaluate
├── train.py                      # Phase 3 baseline → models/model.pkl
├── registry.py                   # MLflow names, require_tracking_uri, pick_best, register_and_promote
├── track_experiments.py          # CONFIGS (5 runs), train_and_log
├── orchestrate_training.py       # Prefect flow
├── schemas.py                    # Listing, PricePrediction, EXAMPLE_LISTING
├── main.py                       # FastAPI app
├── scripts/__init__.py
├── scripts/make_sample.py        # writes tests/fixtures/listings_sample.csv
├── scripts/ci_seed_model.py      # CI: train baseline + promote to @champion
├── scripts/trigger_deploy.py     # POSTs workflow_dispatch for deploy.yml
└── tests/
    ├── conftest.py               # sample_splits, local_mlflow fixtures
    ├── fixtures/listings_sample.csv
    ├── test_features.py
    ├── test_schemas.py
    ├── test_api.py
    ├── test_track_experiments.py
    ├── test_trigger_deploy.py
    └── test_model_registry.py    # real registry URI; skipped if MLFLOW_TRACKING_URI unset
```

Scripts inside `scripts/` import root modules, so always run them as modules: `python -m scripts.<name>`.

---

### Task 1: Project scaffold, Git, venv, pinned requirements (Phase 1)

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pytest.ini`, `README.md`

**Interfaces:**
- Produces: an activated `.venv` (Python 3.11) with all deps; `pytest` discovers `tests/` and can import root modules.

- [ ] **Step 1: Initialise Git on `main`**

```bash
git init -b main
```

- [ ] **Step 2: Create a Python 3.11 venv with uv** (the Anaconda base env has mismatched numexpr/bottleneck; don't use it)

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version   # Expected: Python 3.11.x
```

- [ ] **Step 3: Install unpinned deps, then pin to what was actually installed**

```bash
uv pip install scikit-learn pandas numpy joblib fastapi uvicorn pydantic mlflow prefect pytest requests httpx
uv pip freeze | grep -iE '^(scikit-learn|pandas|numpy|joblib|fastapi|uvicorn|pydantic|mlflow|prefect|pytest|requests|httpx)==' > requirements.txt
cat requirements.txt
```
Expected: exactly 12 pinned lines (resolved on 2026-09-24 as e.g. `mlflow==3.16.1`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `prefect==3.8.6`, `fastapi==0.141.1` — use whatever your freeze prints).

- [ ] **Step 4: Install DVC into the venv (dev tool, deliberately not in requirements.txt — CI doesn't need it)**

```bash
uv pip install dvc
dvc --version
```

- [ ] **Step 5: Write `.gitignore`**

```gitignore
# Python
.venv/
__pycache__/
*.pyc
.pytest_cache/

# Local artifacts (models live in the MLflow registry, data lives in DVC)
models/
mlruns/
mlartifacts/
mlflow.db
mlflow.log

# Misc
.DS_Store
.env

# NOTE: do NOT add data/ here. `dvc add` writes data/.gitignore for the CSV,
# and ignoring the whole folder would stop git from tracking the .dvc pointer file.
```

- [ ] **Step 6: Write `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 7: Write a stub `README.md`**

```markdown
# NYC Airbnb Price Prediction — MLOps practice project

Predicts a NYC Airbnb listing's nightly price (USD). See
`Implementation_Plan_NYC_Airbnb_Price_Prediction.md` for the spec and
`docs/superpowers/plans/` for the build plan.

## Setup
    uv venv --python 3.11 .venv && source .venv/bin/activate
    uv pip install -r requirements.txt dvc
    dvc pull
```

- [ ] **Step 8: Commit**

```bash
git add .gitignore requirements.txt pytest.ini README.md Implementation_Plan_NYC_Airbnb_Price_Prediction.md docs/
git commit -m "chore: project scaffold, pinned requirements, pytest config"
```

---

### Task 2: Version the dataset with DVC (Phase 2)

**Files:**
- Create: `data/AB_NYC_2019.csv` (DVC-tracked), `data/AB_NYC_2019.csv.dvc`, `data/.gitignore`, `.dvc/`, `.dvcignore`

**Interfaces:**
- Produces: `data/AB_NYC_2019.csv` on disk; `dvc pull` restores it from `~/dvc-storage/nyc-airbnb-price`.

- [ ] **Step 1: Copy the dataset in under the canonical name and confirm its shape**

```bash
mkdir -p data
cp ~/Downloads/"Airbnb NYC 2019.csv" data/AB_NYC_2019.csv
python -c "import pandas as pd; df = pd.read_csv('data/AB_NYC_2019.csv'); print(df.shape); print(list(df.columns))"
```
Expected: `(48895, 16)` and the 16 columns from spec §2.

- [ ] **Step 2: `dvc init`, then `dvc add` — BEFORE any `git add -A`** (the Article 7.5 ordering lesson)

```bash
dvc init
dvc add data/AB_NYC_2019.csv
git status --short
```
Expected: `data/AB_NYC_2019.csv.dvc` and `data/.gitignore` appear as new; `data/AB_NYC_2019.csv` itself does **not** appear. If the CSV appears, stop — something is wrong with ignore rules.

- [ ] **Step 3: Configure a local remote outside the repo and push**

```bash
mkdir -p ~/dvc-storage/nyc-airbnb-price
dvc remote add -d localremote ~/dvc-storage/nyc-airbnb-price
dvc push
```
Expected: `1 file pushed`.

- [ ] **Step 4: Prove the round trip**

```bash
rm data/AB_NYC_2019.csv
dvc pull
wc -l data/AB_NYC_2019.csv
```
Expected: `49081 data/AB_NYC_2019.csv` (more lines than rows because some `name` fields contain newlines).

- [ ] **Step 5: Commit**

```bash
git add .dvc .dvcignore data/AB_NYC_2019.csv.dvc data/.gitignore
git commit -m "data: track AB_NYC_2019.csv with DVC and a local remote"
```

---

### Task 3: Shared feature/pipeline module + CI sample (Phase 3 core)

**Files:**
- Create: `features.py`, `scripts/__init__.py`, `scripts/make_sample.py`, `tests/conftest.py`, `tests/fixtures/listings_sample.csv`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces (in `features.py`):
  - `DEFAULT_DATA_PATH: Path`, `TARGET = "price"`, `NUMERIC_FEATURES: list[str]`, `CATEGORICAL_FEATURES: list[str]`, `FEATURES: list[str]`, `MAX_PRICE = 800`, `RANDOM_STATE = 42`
  - `load_data(path: str | Path | None = None) -> pd.DataFrame` — `None` → `$DATA_PATH` or `DEFAULT_DATA_PATH`
  - `clean_data(df) -> pd.DataFrame` — columns exactly `FEATURES + [TARGET]`
  - `split_data(df) -> (X_train, X_test, y_train, y_test)` — 80/20, `random_state=42`
  - `build_model(regressor) -> TransformedTargetRegressor` — `.predict()` returns USD
  - `evaluate(model, X, y) -> dict[str, float]` with keys `rmse`, `mae`, `r2`
- Produces (in `tests/conftest.py`): session fixture `sample_splits` → `split_data(clean_data(load_data(sample)))`

- [ ] **Step 1: Write the failing tests** — `tests/test_features.py`

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import features


@pytest.fixture
def raw_df():
    """Six raw listings with all 16 CSV columns and the real quirks from spec §2."""
    base = dict(
        id=1, name="x", host_id=1, host_name="h",
        neighbourhood_group="Manhattan", neighbourhood="Midtown",
        latitude=40.75, longitude=-73.98, room_type="Entire home/apt",
        minimum_nights=1, number_of_reviews=10, last_review="2019-05-01",
        reviews_per_month=1.5, calculated_host_listings_count=1, availability_365=100,
    )
    prices = [0, 50, 150, 300, 800, 10000]
    rows = [{**base, "id": i, "price": p} for i, p in enumerate(prices)]
    # A listing with no reviews has no review rate (and no last_review).
    rows[1].update(number_of_reviews=0, reviews_per_month=np.nan, last_review=np.nan)
    return pd.DataFrame(rows)


def test_clean_data_drops_zero_and_outlier_prices(raw_df):
    cleaned = features.clean_data(raw_df)
    assert cleaned["price"].tolist() == [50, 150, 300, 800]


def test_clean_data_fills_missing_reviews_per_month_with_zero(raw_df):
    cleaned = features.clean_data(raw_df)
    assert cleaned["reviews_per_month"].isna().sum() == 0
    assert cleaned.loc[cleaned["price"] == 50, "reviews_per_month"].item() == 0.0


def test_clean_data_keeps_only_model_columns(raw_df):
    cleaned = features.clean_data(raw_df)
    assert list(cleaned.columns) == features.FEATURES + ["price"]
    for dropped in ["id", "name", "host_id", "host_name", "last_review"]:
        assert dropped not in cleaned.columns


def test_log_target_inversion_matches_hand_computed_value(raw_df):
    cleaned = features.clean_data(raw_df)
    X, y = cleaned[features.FEATURES], cleaned["price"]
    model = features.build_model(DummyRegressor(strategy="mean")).fit(X, y)

    # Hand computation: the dummy learns mean(log1p(price)); predict() must
    # undo the log with expm1, giving back dollars.
    expected = np.expm1(np.mean(np.log1p([50, 150, 300, 800])))
    assert model.predict(X.iloc[[0]])[0] == pytest.approx(expected)
    # If the inversion were missing we'd get ~5.3 (a log value), not ~$206.
    assert 150 < expected < 325


def test_evaluate_returns_dollar_scale_metrics(raw_df):
    cleaned = features.clean_data(raw_df)
    X, y = cleaned[features.FEATURES], cleaned["price"]
    model = features.build_model(DummyRegressor(strategy="mean")).fit(X, y)
    metrics = features.evaluate(model, X, y)
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] > 100  # errors are in dollars, not log units
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_features.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'features'`.

- [ ] **Step 3: Implement `features.py`**

```python
"""Shared data loading, cleaning and model-building code.

Every training entry point (train.py, track_experiments.py,
orchestrate_training.py, scripts/ci_seed_model.py) imports from here so the
cleaning rules and preprocessing can never drift apart.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATA_PATH = Path("data/AB_NYC_2019.csv")

TARGET = "price"
NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]
# neighbourhood has 221 distinct values: a high-cardinality categorical.
# One-hot still works; handle_unknown="ignore" zeroes any unseen value at predict time.
CATEGORICAL_FEATURES = ["neighbourhood_group", "neighbourhood", "room_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Cleaning thresholds (spec §2 / Phase 3):
# - price == 0: 11 rows in the 2019 file, data errors rather than free listings.
# - price > 800: ~the 99th percentile ($799); 420 extreme outliers up to $10,000.
MAX_PRICE = 800
RANDOM_STATE = 42


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSV. With no path, honour $DATA_PATH (CI uses the sample)."""
    path = path or os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # reviews_per_month is missing exactly when number_of_reviews == 0:
    # no reviews means a review rate of 0, not the average rate.
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)
    df = df[(df[TARGET] > 0) & (df[TARGET] <= MAX_PRICE)]
    return df[FEATURES + [TARGET]].reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """80/20 split -> (X_train, X_test, y_train, y_test)."""
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=RANDOM_STATE
    )


def build_model(regressor) -> TransformedTargetRegressor:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])
    # Price is heavily right-skewed, so the regressor is fit on log1p(price).
    # TransformedTargetRegressor applies expm1 inside .predict(), so every
    # caller (evaluation, the API, tests) gets dollars back automatically.
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1
    )


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """RMSE / MAE / R² on the original dollar scale."""
    predictions = model.predict(X)
    return {
        "rmse": float(root_mean_squared_error(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: 5 passed.

- [ ] **Step 5: Write `scripts/__init__.py` (empty) and `scripts/make_sample.py`**

```python
"""Write the small, PII-free sample used by tests and CI.

CI can't `dvc pull` (the DVC remote lives on this laptop), so it trains on
this committed sample instead. Only model columns are kept: no names or ids.
"""
from pathlib import Path

from features import DEFAULT_DATA_PATH, FEATURES, RANDOM_STATE, TARGET, load_data

SAMPLE_PATH = Path("tests/fixtures/listings_sample.csv")
SAMPLE_SIZE = 2000


def main():
    df = load_data(DEFAULT_DATA_PATH)
    sample = df[FEATURES + [TARGET]].sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"wrote {len(sample)} rows to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
```

Run:
```bash
touch scripts/__init__.py
python -m scripts.make_sample
python -c "import pandas as pd; s = pd.read_csv('tests/fixtures/listings_sample.csv'); print(s.shape); print(s.neighbourhood_group.value_counts())"
```
Expected: `wrote 2000 rows ...`, shape `(2000, 11)`, all 5 boroughs present.

- [ ] **Step 6: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest

from features import clean_data, load_data, split_data

SAMPLE_PATH = Path(__file__).parent / "fixtures" / "listings_sample.csv"


@pytest.fixture(scope="session")
def sample_path():
    """Absolute path, so tests pass no matter which directory pytest runs from."""
    return SAMPLE_PATH


@pytest.fixture(scope="session")
def sample_splits(sample_path):
    """(X_train, X_test, y_train, y_test) from the committed 2,000-row sample."""
    return split_data(clean_data(load_data(sample_path)))
```

- [ ] **Step 7: Add split + unseen-category tests to `tests/test_features.py`, run, confirm pass**

```python
def test_split_data_is_reproducible_80_20(sample_splits, sample_path):
    X_train, X_test, y_train, y_test = sample_splits
    assert len(X_test) / (len(X_train) + len(X_test)) == pytest.approx(0.2, abs=0.01)
    assert list(X_train.columns) == features.FEATURES
    again = features.split_data(features.clean_data(features.load_data(sample_path)))
    assert again[0].index.equals(X_train.index)


def test_model_tolerates_unseen_neighbourhood(raw_df):
    # 221 neighbourhoods means the API will eventually receive one the model
    # never saw; handle_unknown="ignore" must turn that into zeros, not a crash.
    cleaned = features.clean_data(raw_df)
    X, y = cleaned[features.FEATURES], cleaned["price"]
    model = features.build_model(DummyRegressor(strategy="mean")).fit(X, y)
    unseen = X.iloc[[0]].assign(neighbourhood="Nowhere Heights")
    assert model.predict(unseen)[0] > 0
```

Run: `pytest tests/test_features.py -v` → Expected: 7 passed.

- [ ] **Step 8: Commit**

```bash
git add features.py scripts/__init__.py scripts/make_sample.py tests/
git commit -m "feat: shared cleaning/pipeline module with log-target model and CI sample"
```

---

### Task 4: Baseline training script (Phase 3)

**Files:**
- Create: `train.py`

**Interfaces:**
- Consumes: `load_data`, `clean_data`, `split_data`, `build_model`, `evaluate` from `features.py`
- Produces: `models/model.pkl` (joblib, git-ignored) — a sanity artifact only; serving uses the registry.

- [ ] **Step 1: Write `train.py`**

```python
"""Phase 3 baseline: LinearRegression trained on log1p(price), scored in dollars.

features.build_model wraps the pipeline in a TransformedTargetRegressor that
fits on log1p(price) and applies expm1 on predict, so the metrics printed
here are real USD errors. If RMSE ever comes out in the thousands (or below
1), the log/exp inversion is the first thing to check.
"""
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from features import build_model, clean_data, evaluate, load_data, split_data

MODEL_PATH = Path("models/model.pkl")


def main():
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"rows after cleaning: {len(df)}  (train={len(X_train)}, test={len(X_test)})")

    model = build_model(LinearRegression()).fit(X_train, y_train)
    for name, value in evaluate(model, X_test, y_test).items():
        print(f"{name}: {value:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    assert (reloaded.predict(X_test.head()) == model.predict(X_test.head())).all()
    print(f"saved and reloaded {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and check the numbers are sane**

Run: `python train.py`
Expected (measured on 2026-09-24 with sklearn 1.5; small differences are fine):
```
rows after cleaning: 48464  (train=38771, test=9693)
rmse: ~83.5
mae: ~47.1
r2: ~0.395
saved and reloaded models/model.pkl
```
Tens of dollars = correct. If RMSE is ~thousands or R² is hugely negative, stop and debug the inversion.

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: baseline LinearRegression training script (log-price target)"
```

---

### Task 5: Pydantic schemas (Phase 4a + Phase 9 schema tests)

**Files:**
- Create: `schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `Listing` (fields == `features.FEATURES`), `PricePrediction(predicted_price: float, currency: str = "USD")`, `EXAMPLE_LISTING: dict` (a valid Midtown entire-home listing), `NYC_LAT_MIN/MAX`, `NYC_LON_MIN/MAX`.

- [ ] **Step 1: Write the failing tests** — `tests/test_schemas.py`

```python
import pytest
from pydantic import ValidationError

import features
from schemas import EXAMPLE_LISTING, Listing, PricePrediction


def test_valid_listing_is_accepted():
    listing = Listing(**EXAMPLE_LISTING)
    assert listing.room_type == "Entire home/apt"


def test_listing_fields_match_model_features_exactly():
    assert set(Listing.model_fields) == set(features.FEATURES)


@pytest.mark.parametrize(
    "field, value",
    [
        ("room_type", "Castle"),
        ("neighbourhood_group", "New Jersey"),
        ("neighbourhood", ""),
        ("minimum_nights", 0),
        ("number_of_reviews", -1),
        ("reviews_per_month", -0.5),
        ("calculated_host_listings_count", 0),
        ("availability_365", 366),
        ("latitude", -75.0),    # Antarctica
        ("longitude", 73.98),   # sign flipped: that's China
    ],
)
def test_invalid_value_is_rejected(field, value):
    with pytest.raises(ValidationError):
        Listing(**{**EXAMPLE_LISTING, field: value})


def test_missing_field_is_rejected():
    incomplete = {k: v for k, v in EXAMPLE_LISTING.items() if k != "room_type"}
    with pytest.raises(ValidationError):
        Listing(**incomplete)


def test_price_prediction_defaults_to_usd():
    assert PricePrediction(predicted_price=120.5).currency == "USD"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_schemas.py -v`
Expected: ERROR — `No module named 'schemas'`.

- [ ] **Step 3: Implement `schemas.py`**

```python
"""Request/response models for the price API.

Fields mirror the model's feature set (features.FEATURES), not the raw CSV:
no id, name, host or last_review.
"""
from typing import Literal

from pydantic import BaseModel, Field

# NYC bounding box, padded slightly around the 2019 data's observed range
# (lat 40.4998–40.9131, lon -74.2444 to -73.7130). A listing claiming to be
# in Antarctica should never reach the model.
NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
NYC_LON_MIN, NYC_LON_MAX = -74.26, -73.70

EXAMPLE_LISTING = {
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Midtown",
    "latitude": 40.7549,
    "longitude": -73.9840,
    "room_type": "Entire home/apt",
    "minimum_nights": 2,
    "number_of_reviews": 20,
    "reviews_per_month": 1.0,
    "calculated_host_listings_count": 1,
    "availability_365": 180,
}


class Listing(BaseModel):
    model_config = {"json_schema_extra": {"examples": [EXAMPLE_LISTING]}}

    neighbourhood_group: Literal["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    # 221 values in training data: too many for a Literal. Unknown names are
    # accepted and zeroed by the model's OneHotEncoder(handle_unknown="ignore").
    neighbourhood: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=NYC_LAT_MIN, le=NYC_LAT_MAX)
    longitude: float = Field(..., ge=NYC_LON_MIN, le=NYC_LON_MAX)
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: 14 passed.

- [ ] **Step 5: Break a constraint on purpose and watch it fail** (spec §5, Phase 9 DoD)

Temporarily change `minimum_nights: int = Field(..., ge=1)` to `Field(..., ge=0)`.
Run: `pytest tests/test_schemas.py -v`
Expected: exactly `test_invalid_value_is_rejected[minimum_nights-0]` FAILS with `DID NOT RAISE`. Revert the change and rerun → 14 passed.

- [ ] **Step 6: Commit**

```bash
git add schemas.py tests/test_schemas.py
git commit -m "feat: Listing/PricePrediction schemas with NYC bounds and tests"
```

---

### Task 6: FastAPI service (Phase 4b)

**Files:**
- Create: `main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Listing`, `PricePrediction`, `EXAMPLE_LISTING` from `schemas.py`
- Produces: `main.app` (FastAPI), `main.load_model() -> model` (monkeypatchable), `main.MODEL_URI` (env `MODEL_URI`, default `models:/AirbnbPriceModel@champion`). Endpoints: `GET /health` → `{"status": "ok", "model_uri": str}`; `POST /predict` → `PricePrediction`.

- [ ] **Step 1: Write the failing tests** — `tests/test_api.py`

```python
import numpy as np
import pytest
from fastapi.testclient import TestClient

import features
import main
from schemas import EXAMPLE_LISTING


class FakeModel:
    """Stands in for the registry model so API tests need no MLflow server."""

    def __init__(self, price):
        self.price = price
        self.seen = None

    def predict(self, X):
        self.seen = X
        return np.array([self.price])


@pytest.fixture
def fake_model(monkeypatch):
    fake = FakeModel(price=123.456)
    monkeypatch.setattr(main, "load_model", lambda: fake)
    return fake


@pytest.fixture
def client(fake_model):
    with TestClient(main.app) as c:  # `with` runs the lifespan (model load)
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_rounded_usd_price(client):
    response = client.post("/predict", json=EXAMPLE_LISTING)
    assert response.status_code == 200
    assert response.json() == {"predicted_price": 123.46, "currency": "USD"}


def test_predict_sends_exactly_the_model_features(client, fake_model):
    client.post("/predict", json=EXAMPLE_LISTING)
    assert set(fake_model.seen.columns) == set(features.FEATURES)
    assert len(fake_model.seen) == 1


def test_predict_never_returns_negative_price(client, fake_model):
    fake_model.price = -5.0
    assert client.post("/predict", json=EXAMPLE_LISTING).json()["predicted_price"] == 0.0


def test_predict_rejects_invalid_listing(client):
    response = client.post("/predict", json={**EXAMPLE_LISTING, "room_type": "Castle"})
    assert response.status_code == 422
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api.py -v`
Expected: ERROR — `No module named 'main'`.

- [ ] **Step 3: Implement `main.py`**

```python
"""FastAPI service for AirbnbPriceModel@champion.

The model is loaded once at startup from the MLflow registry. mlflow reads
MLFLOW_TRACKING_URI from the environment, so the same image works against a
laptop server, CI, or the Compose `mlflow-server` service.
"""
import os
from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI

from schemas import Listing, PricePrediction

MODEL_URI = os.environ.get("MODEL_URI", "models:/AirbnbPriceModel@champion")


def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="NYC Airbnb Price API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PricePrediction)
def predict(listing: Listing) -> PricePrediction:
    features = pd.DataFrame([listing.model_dump()])
    # The champion is a TransformedTargetRegressor: .predict() already applies
    # expm1 to undo the log1p it was trained on, so this value is in USD.
    price = float(app.state.model.predict(features)[0])
    return PricePrediction(predicted_price=round(max(price, 0.0), 2))
```

- [ ] **Step 4: Run all tests**

Run: `pytest -v`
Expected: 26 passed (7 features + 14 schemas + 5 api).

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: FastAPI app serving the registry champion"
```

---

### Task 7: MLflow server + experiment tracking (Phase 6)

**Files:**
- Create: `registry.py` (constants + `require_tracking_uri` only in this task), `track_experiments.py`
- Modify: `tests/conftest.py` (add `local_mlflow` fixture)
- Test: `tests/test_track_experiments.py`

**Interfaces:**
- Produces (`registry.py`): `EXPERIMENT_NAME = "airbnb-price-prediction"`, `MODEL_NAME = "AirbnbPriceModel"`, `CHAMPION_ALIAS = "champion"`, `require_tracking_uri() -> str` (exits with a helpful message if unset).
- Produces (`track_experiments.py`): `CONFIGS: dict[str, tuple[type, dict]]` (5 run names from spec Phase 6), `train_and_log(run_name, X_train, X_test, y_train, y_test) -> tuple[str, dict[str, float]]` (run_id, metrics). The model is logged under artifact name `model`, so its URI is `runs:/<run_id>/model`.
- Produces (`conftest.py`): `local_mlflow` fixture — throwaway sqlite tracking store + registry in `tmp_path`, with an active experiment.

- [ ] **Step 1: Start the MLflow server (separate terminal, leave it running)**

First check port 5000 — on macOS the AirPlay Receiver often holds it:
```bash
lsof -nP -iTCP:5000 -sTCP:LISTEN
```
If `ControlCenter` is listed, turn off System Settings → General → AirDrop & Handoff → AirPlay Receiver (or use port 5001 everywhere below).

Check whether this MLflow version has host-header validation:
```bash
mlflow server --help | grep -A3 -- --allowed-hosts
```
If the flag exists, include it (Docker needs `host.docker.internal`; Compose needs `mlflow-server`):
```bash
source .venv/bin/activate
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*,mlflow-server:*"
```
(Drop the last line if the flag doesn't exist.) `--artifacts-destination` makes the server proxy artifacts over HTTP, which is what lets a Docker container download the model later.

In your working terminal:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
curl -s $MLFLOW_TRACKING_URI/health   # Expected: OK
```

- [ ] **Step 2: Add the `local_mlflow` fixture to `tests/conftest.py`**

Append:
```python
import mlflow


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """A throwaway sqlite tracking store + registry, so unit tests never touch the real server."""
    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        "test-experiment", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    yield uri
    mlflow.set_tracking_uri(previous_uri)
```
(Move `import mlflow` to the top of the file with the other imports.)

- [ ] **Step 3: Write the failing test** — `tests/test_track_experiments.py`

```python
import mlflow
import pytest

from track_experiments import CONFIGS, train_and_log


def test_configs_match_the_spec():
    assert list(CONFIGS) == [
        "linreg_baseline", "rf_100", "rf_300_depth10", "gb_100_lr01", "gb_200_lr005",
    ]


def test_train_and_log_records_params_metrics_and_model(local_mlflow, sample_splits):
    run_id, metrics = train_and_log("linreg_baseline", *sample_splits)

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "linreg_baseline"
    assert run.data.params["model_type"] == "LinearRegression"
    assert run.data.params["target_transform"] == "log1p/expm1"
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert run.data.metrics["rmse"] == pytest.approx(metrics["rmse"])

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert (model.predict(sample_splits[1]) > 0).all()
```

- [ ] **Step 4: Run to verify failure**

Run: `pytest tests/test_track_experiments.py -v`
Expected: ERROR — `No module named 'track_experiments'`.

- [ ] **Step 5: Write `registry.py` (naming constants for now; promotion comes in Task 8)**

```python
"""MLflow naming and Model Registry helpers.

Uses registered-model aliases (@champion), not the deprecated stages API.
"""
import os
import sys

EXPERIMENT_NAME = "airbnb-price-prediction"
MODEL_NAME = "AirbnbPriceModel"
CHAMPION_ALIAS = "champion"


def require_tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        sys.exit(
            "MLFLOW_TRACKING_URI is not set. Start the server (see README) and run:\n"
            "  export MLFLOW_TRACKING_URI=http://127.0.0.1:5000"
        )
    return uri
```

- [ ] **Step 6: Write `track_experiments.py`**

```python
"""Phase 6: train the five candidate configs and log each as an MLflow run.

Metrics are on the dollar scale (features.evaluate), and every logged model
carries its own log1p/expm1 target transform.
"""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from features import RANDOM_STATE, build_model, clean_data, evaluate, load_data, split_data
from registry import EXPERIMENT_NAME, require_tracking_uri

# run name -> (model class, params). Params are logged to MLflow verbatim.
CONFIGS = {
    "linreg_baseline": (LinearRegression, {}),
    "rf_100": (RandomForestRegressor, {"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_STATE}),
    "rf_300_depth10": (RandomForestRegressor, {"n_estimators": 300, "max_depth": 10, "n_jobs": -1, "random_state": RANDOM_STATE}),
    "gb_100_lr01": (GradientBoostingRegressor, {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE}),
    "gb_200_lr005": (GradientBoostingRegressor, {"n_estimators": 200, "learning_rate": 0.05, "random_state": RANDOM_STATE}),
}


def train_and_log(run_name, X_train, X_test, y_train, y_test):
    """Fit one config inside an MLflow run. Returns (run_id, metrics)."""
    model_class, params = CONFIGS[run_name]
    with mlflow.start_run(run_name=run_name) as run:
        model = build_model(model_class(**params)).fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_params(
            {"model_type": model_class.__name__, "target_transform": "log1p/expm1", **params}
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3))
    return run.info.run_id, metrics


def main():
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    splits = split_data(clean_data(load_data()))
    for run_name in CONFIGS:
        run_id, m = train_and_log(run_name, *splits)
        print(f"{run_name:16s} rmse={m['rmse']:7.2f}  mae={m['mae']:6.2f}  r2={m['r2']:.3f}  run_id={run_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the tests**

Run: `pytest tests/test_track_experiments.py -v`
Expected: 2 passed.
Troubleshooting: if `log_model` errors on `input_example` schema inference (pandas 3 string dtype), confirm with `python -c "import mlflow, pandas; print(mlflow.__version__, pandas.__version__)"` and drop `input_example=` rather than downgrading pandas. If `name=` is rejected, the installed MLflow is 2.x — use `artifact_path="model"` instead.

- [ ] **Step 8: Run all 5 configs against the real server**

Run: `python track_experiments.py` (RF/GB runs take a few minutes).
Expected: 5 lines; linreg rmse ≈ 83.5; tree models should beat it.
Then open http://127.0.0.1:5000 → experiment `airbnb-price-prediction` → confirm **5 runs** with params, `rmse/mae/r2`, and a `model` artifact each (Phase 6 DoD). Write the five metric rows down; they go in the README in Task 14.

- [ ] **Step 9: Commit**

```bash
git add registry.py track_experiments.py tests/conftest.py tests/test_track_experiments.py
git commit -m "feat: MLflow experiment tracking for five regression configs"
```

---

### Task 8: Model Registry — pick best, register, promote to @champion (Phase 7 + registry tests)

**Files:**
- Modify: `registry.py` (add `pick_best`, `best_run_id`, `register_and_promote`, `__main__`)
- Test: `tests/test_track_experiments.py` (add promotion tests), `tests/test_model_registry.py`

**Interfaces:**
- Consumes: `train_and_log` (Task 7), `local_mlflow`, `sample_splits` fixtures
- Produces:
  - `SELECTION_METRIC = "rmse"`
  - `pick_best(results: dict[str, dict[str, float]]) -> str` — maps run_id → metrics; returns lowest-RMSE run_id; prints a note if lowest-MAE differs
  - `best_run_id(experiment_name: str) -> str` — over FINISHED runs of the experiment
  - `register_and_promote(run_id: str) -> str` — registers `runs:/<run_id>/model` as `AirbnbPriceModel`, sets `@champion`, returns the version string

- [ ] **Step 1: Write failing tests** — append to `tests/test_track_experiments.py`

```python
from mlflow import MlflowClient

from registry import CHAMPION_ALIAS, MODEL_NAME, pick_best, register_and_promote


def test_pick_best_uses_lowest_rmse():
    results = {
        "a": {"rmse": 90.0, "mae": 50.0, "r2": 0.30},
        "b": {"rmse": 70.0, "mae": 45.0, "r2": 0.50},
        "c": {"rmse": 80.0, "mae": 40.0, "r2": 0.45},
    }
    assert pick_best(results) == "b"  # RMSE decides even though "c" has the lowest MAE


def test_register_and_promote_sets_champion_alias(local_mlflow, sample_splits):
    run_id, _ = train_and_log("linreg_baseline", *sample_splits)
    version = register_and_promote(run_id)

    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert champion.version == version
    assert champion.run_id == run_id

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    assert (model.predict(sample_splits[1]) > 0).all()


def test_promoting_again_moves_the_alias(local_mlflow, sample_splits):
    first = register_and_promote(train_and_log("linreg_baseline", *sample_splits)[0])
    second = register_and_promote(train_and_log("linreg_baseline", *sample_splits)[0])
    assert int(second) == int(first) + 1
    assert MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version == second
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_track_experiments.py -v`
Expected: ImportError — `cannot import name 'pick_best' from 'registry'`.

- [ ] **Step 3: Extend `registry.py`**

Add imports at the top:
```python
import mlflow
from mlflow import MlflowClient
```
Append:
```python
# Selection rule: lowest RMSE on the held-out test set wins. RMSE is the
# deciding metric because it punishes large dollar misses hardest — for a
# pricing tool, one $300 miss hurts more than three $100 misses. MAE is
# checked too, so a disagreement is printed rather than silently ignored.
SELECTION_METRIC = "rmse"


def pick_best(results: dict[str, dict[str, float]]) -> str:
    """results maps run_id -> metrics. Returns the run_id with the lowest RMSE."""
    best = min(results, key=lambda run_id: results[run_id][SELECTION_METRIC])
    best_by_mae = min(results, key=lambda run_id: results[run_id]["mae"])
    if best_by_mae != best:
        print(f"note: lowest-RMSE run {best} differs from lowest-MAE run {best_by_mae}; RMSE decides")
    return best


def best_run_id(experiment_name: str) -> str:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'",
    )
    results = {
        row["run_id"]: {"rmse": row["metrics.rmse"], "mae": row["metrics.mae"], "r2": row["metrics.r2"]}
        for _, row in runs.iterrows()
    }
    return pick_best(results)


def register_and_promote(run_id: str) -> str:
    """Register the run's model and point @champion at it. Returns the new version."""
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    return version


if __name__ == "__main__":
    require_tracking_uri()
    run_id = best_run_id(EXPERIMENT_NAME)
    version = register_and_promote(run_id)
    print(f"registered {MODEL_NAME} v{version} from run {run_id} -> @{CHAMPION_ALIAS}")
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_track_experiments.py -v`
Expected: 5 passed.

- [ ] **Step 5: Promote the real best run**

```bash
python registry.py
```
Expected: `registered AirbnbPriceModel v1 from run <id> -> @champion` (plus a `note:` line if MAE disagrees — record which one won and why in the README later). In the UI → Models → `AirbnbPriceModel` shows version 1 with alias `champion`.

- [ ] **Step 6: Write `tests/test_model_registry.py`** (real registry, spec Phase 9)

```python
"""Loads the real champion via the registry URI (not joblib) and checks that
its predictions make economic sense. Regression has no 0.5 threshold to
assert against, so we check direction and plausibility instead.

Needs MLFLOW_TRACKING_URI pointing at a server where
AirbnbPriceModel@champion exists (locally after Task 8; in CI after
scripts/ci_seed_model.py).
"""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest

from features import MAX_PRICE
from registry import CHAMPION_ALIAS, MODEL_NAME
from schemas import EXAMPLE_LISTING, Listing

pytestmark = pytest.mark.skipif(
    not os.environ.get("MLFLOW_TRACKING_URI"),
    reason="MLFLOW_TRACKING_URI not set; start the MLflow server to run registry tests",
)

MANHATTAN_ENTIRE_HOME = EXAMPLE_LISTING
BRONX_SHARED_ROOM = {
    **EXAMPLE_LISTING,
    "neighbourhood_group": "Bronx",
    "neighbourhood": "Fordham",
    "latitude": 40.8615,
    "longitude": -73.8904,
    "room_type": "Shared room",
}


@pytest.fixture(scope="module")
def champion():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")


def predict(model, listing: dict) -> float:
    Listing(**listing)  # the test inputs must be valid API inputs too
    return float(model.predict(pd.DataFrame([listing]))[0])


def test_manhattan_entire_home_costs_more_than_bronx_shared_room(champion):
    assert predict(champion, MANHATTAN_ENTIRE_HOME) > predict(champion, BRONX_SHARED_ROOM)


@pytest.mark.parametrize("listing", [MANHATTAN_ENTIRE_HOME, BRONX_SHARED_ROOM])
def test_predictions_are_plausible_dollar_amounts(champion, listing):
    assert 10 < predict(champion, listing) < MAX_PRICE
```

- [ ] **Step 7: Run it against the real server**

Run: `pytest tests/test_model_registry.py -v` (with `MLFLOW_TRACKING_URI` exported)
Expected: 3 passed. Then `env -u MLFLOW_TRACKING_URI pytest tests/test_model_registry.py -v` → 3 skipped with the reason message.

- [ ] **Step 8: Phase 7 DoD — load the champion from a separate process**

```bash
python -c "
import mlflow.sklearn, pandas as pd
from schemas import EXAMPLE_LISTING
m = mlflow.sklearn.load_model('models:/AirbnbPriceModel@champion')
print(round(float(m.predict(pd.DataFrame([EXAMPLE_LISTING]))[0]), 2))"
```
Expected: a single plausible Midtown entire-home price (roughly $150–$300).

- [ ] **Step 9: Serve locally and hit it**

```bash
uvicorn main:app --port 8000 &
sleep 5
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"neighbourhood_group":"Manhattan","neighbourhood":"Midtown","latitude":40.7549,"longitude":-73.984,"room_type":"Entire home/apt","minimum_nights":2,"number_of_reviews":20,"reviews_per_month":1.0,"calculated_host_listings_count":1,"availability_365":180}'
kill %1
```
Expected: `{"status":"ok",...}` then `{"predicted_price":<same as Step 8>,"currency":"USD"}`.

- [ ] **Step 10: Commit**

```bash
git add registry.py tests/test_track_experiments.py tests/test_model_registry.py
git commit -m "feat: register best run as AirbnbPriceModel@champion with registry tests"
```

---

### Task 9: Docker image for the API (Phase 5)

**Files:**
- Create: `requirements-serve.txt`, `Dockerfile`, `.dockerignore`

**Interfaces:**
- Consumes: `main.py`, `schemas.py`; a running MLflow server with `@champion`
- Produces: image `airbnb-price-api` listening on 8000; requires env `MLFLOW_TRACKING_URI` at run time.

**Decision (spec asks to document it):** the model is **not** baked into the image and not trained during build. The container pulls `@champion` from the registry at startup. Why: one image serves any promoted version (retraining never requires a rebuild), the build stays fast and deterministic, and it matches Phase 4's "registry URI, not a hardcoded path" rule. Cost: the container needs network access to the MLflow server at startup — acceptable here, and made explicit by Compose in Task 13.

- [ ] **Step 1: Write `requirements-serve.txt`** using the *same* versions as `requirements.txt` (sklearn must match between training and serving or unpickling can break)

```bash
grep -iE '^(fastapi|uvicorn|pydantic|scikit-learn|pandas|numpy)==' requirements.txt > requirements-serve.txt
echo "mlflow-skinny==$(python -c 'import mlflow; print(mlflow.__version__)')" >> requirements-serve.txt
cat requirements-serve.txt
```
Expected: 7 pinned lines, `mlflow-skinny` matching the `mlflow` version.

- [ ] **Step 2: Write `.dockerignore`**

```
.venv
.git
.dvc/cache
data
models
mlruns
mlartifacts
mlflow.db
mlflow.log
tests
docs
__pycache__
*.pyc
.pytest_cache
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies before code: editing main.py doesn't bust the pip layer cache.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY schemas.py main.py ./

# No model is baked in. main.py loads models:/AirbnbPriceModel@champion from
# MLFLOW_TRACKING_URI at startup, so promoting a new champion only needs a
# container restart. MLFLOW_TRACKING_URI must be supplied at `docker run`.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Build**

Run: `docker build -t airbnb-price-api:local .`
Expected: build succeeds. (If the Docker daemon isn't running, start Docker Desktop first.)

- [ ] **Step 5: Run against the host MLflow server and predict**

```bash
docker run --rm -d --name airbnb-api -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 airbnb-price-api:local
sleep 10
docker logs airbnb-api | tail -5
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"neighbourhood_group":"Manhattan","neighbourhood":"Midtown","latitude":40.7549,"longitude":-73.984,"room_type":"Entire home/apt","minimum_nights":2,"number_of_reviews":20,"reviews_per_month":1.0,"calculated_host_listings_count":1,"availability_365":180}'
docker stop airbnb-api
```
Expected: `Application startup complete` in logs and the same price as Task 8 Step 8.
Troubleshooting: `ModuleNotFoundError` on model load → a dependency is missing from `mlflow-skinny`; add the named package (pinned) to `requirements-serve.txt`, or swap `mlflow-skinny` for `mlflow`. `Invalid Host header`/403 → restart the server with `--allowed-hosts` (Task 7 Step 1). Connection refused → the server must listen on `0.0.0.0`, not `127.0.0.1`.

- [ ] **Step 6: Commit**

```bash
git add requirements-serve.txt Dockerfile .dockerignore
git commit -m "feat: slim Docker image that loads the champion from MLflow at startup"
```

---

### Task 10: Prefect orchestration + deploy trigger (Phase 8)

**Files:**
- Create: `scripts/trigger_deploy.py`, `orchestrate_training.py`
- Test: `tests/test_trigger_deploy.py`

**Interfaces:**
- Consumes: `clean_data`, `load_data`, `split_data`, `CONFIGS`, `train_and_log`, `pick_best`, `register_and_promote`, `require_tracking_uri`, `EXPERIMENT_NAME`
- Produces:
  - `scripts.trigger_deploy.trigger_deploy(model_version: str, repo: str, token: str, ref: str = "main") -> bool`
  - `orchestrate_training.training_flow() -> str` (returns the promoted version). Env: `MLFLOW_TRACKING_URI` (required), `GITHUB_TOKEN` + `GITHUB_REPO` (optional; deploy trigger is skipped with a warning if missing).

- [ ] **Step 1: Write failing tests** — `tests/test_trigger_deploy.py`

```python
import pytest

from scripts import trigger_deploy as td


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_post(url, headers, json, timeout):
        calls.update(url=url, headers=headers, json=json)
        return calls.get("response", FakeResponse(204))

    monkeypatch.setattr(td.requests, "post", fake_post)
    return calls


def test_trigger_deploy_dispatches_deploy_workflow(captured):
    assert td.trigger_deploy("3", repo="me/airbnb", token="tok") is True
    assert captured["url"] == "https://api.github.com/repos/me/airbnb/actions/workflows/deploy.yml/dispatches"
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["json"] == {"ref": "main", "inputs": {"model_version": "3"}}


def test_trigger_deploy_reports_failure(captured):
    captured["response"] = FakeResponse(401, "Bad credentials")
    assert td.trigger_deploy("3", repo="me/airbnb", token="bad") is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_trigger_deploy.py -v`
Expected: ImportError — `cannot import name 'trigger_deploy' from 'scripts'`.

- [ ] **Step 3: Write `scripts/trigger_deploy.py`**

```python
"""Ask GitHub Actions to run deploy.yml (workflow_dispatch) for a promoted model version.

Needs a GitHub token with Actions: write on the repo (fine-grained PAT).
Usage: GITHUB_REPO=owner/repo GITHUB_TOKEN=... python -m scripts.trigger_deploy --model-version 3
"""
import argparse
import os
import sys

import requests

WORKFLOW_FILE = "deploy.yml"


def trigger_deploy(model_version: str, repo: str, token: str, ref: str = "main") -> bool:
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW_FILE}/dispatches",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": ref, "inputs": {"model_version": str(model_version)}},
        timeout=10,
    )
    # GitHub answers 204 (or 200 when returning run details) on success.
    if response.status_code not in (200, 204):
        print(f"deploy trigger failed: {response.status_code} {response.text}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()
    ok = trigger_deploy(args.model_version, os.environ["GITHUB_REPO"], os.environ["GITHUB_TOKEN"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_trigger_deploy.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write `orchestrate_training.py`**

```python
"""Phase 8: Prefect flow — load -> split -> train_and_log x5 -> promote best -> request deploy.

MLFLOW_TRACKING_URI is read from the environment. Run once:
    python orchestrate_training.py
Serve on a weekly schedule (needs `prefect server start` running):
    python orchestrate_training.py --serve
"""
import os
import sys

import mlflow
from prefect import flow, get_run_logger, task

import features
import track_experiments
from registry import EXPERIMENT_NAME, pick_best, register_and_promote, require_tracking_uri
from scripts.trigger_deploy import trigger_deploy


@task(retries=2, retry_delay_seconds=5)
def load_data():
    return features.clean_data(features.load_data())


@task
def split_data(df):
    return features.split_data(df)


@task
def train_and_log(run_name, splits):
    run_id, metrics = track_experiments.train_and_log(run_name, *splits)
    get_run_logger().info("%s rmse=%.2f mae=%.2f r2=%.3f", run_name, metrics["rmse"], metrics["mae"], metrics["r2"])
    return run_id, metrics


@task
def promote_best_model(results):
    run_id = pick_best(results)
    version = register_and_promote(run_id)
    get_run_logger().info("promoted run %s as AirbnbPriceModel v%s @champion", run_id, version)
    return version


@task
def request_deploy(version):
    repo, token = os.environ.get("GITHUB_REPO"), os.environ.get("GITHUB_TOKEN")
    if not (repo and token):
        get_run_logger().warning("GITHUB_REPO/GITHUB_TOKEN not set; skipping deploy trigger")
        return False
    return trigger_deploy(version, repo, token)


@flow(name="airbnb-price-training")
def training_flow():
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    splits = split_data(load_data())
    results = {}
    for run_name in track_experiments.CONFIGS:
        run_id, metrics = train_and_log(run_name, splits)
        results[run_id] = metrics
    version = promote_best_model(results)
    request_deploy(version)
    return version


if __name__ == "__main__":
    if "--serve" in sys.argv:
        training_flow.serve(name="weekly-retrain", cron="0 3 * * 1")  # Mondays 03:00
    else:
        training_flow()
```

- [ ] **Step 6: Manual run first** (spec: schedule only after a manual run works)

Run: `python orchestrate_training.py` (server from Task 7 still running)
Expected: Prefect logs for each task, 5 new runs in the MLflow UI, `AirbnbPriceModel` v2 with `@champion` moved to it, and a warning that the deploy trigger was skipped.

- [ ] **Step 7: Prove the retry works** — temporarily point at a missing file:

```bash
DATA_PATH=nope.csv python orchestrate_training.py
```
Expected: `load_data` fails 3 times (1 try + 2 retries, 5 s apart) then the flow fails. No new MLflow runs.

- [ ] **Step 8: Schedule it**

Terminal A: `prefect server start` → UI at http://127.0.0.1:4200
Terminal B:
```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
python orchestrate_training.py --serve
```
Expected: deployment `airbnb-price-training/weekly-retrain` listed in the Prefect UI with the cron schedule. Trigger a "Quick run" from the UI once to confirm, then Ctrl-C.

- [ ] **Step 9: Commit**

```bash
git add orchestrate_training.py scripts/trigger_deploy.py tests/test_trigger_deploy.py
git commit -m "feat: Prefect training flow with retries, promotion and deploy trigger"
```

---

### Task 11: GitHub repo + CI workflow (Phase 10a)

**Files:**
- Create: `scripts/ci_seed_model.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `train_and_log`, `register_and_promote`, `require_tracking_uri`, `EXPERIMENT_NAME`; `DATA_PATH=tests/fixtures/listings_sample.csv`
- Produces: CI status checks `test` and `build-image` on every PR.

- [ ] **Step 1: Write `scripts/ci_seed_model.py`**

```python
"""CI only: train a quick baseline on the committed sample and make it @champion,
so models:/AirbnbPriceModel@champion resolves for tests/test_model_registry.py.

Run as: DATA_PATH=tests/fixtures/listings_sample.csv python -m scripts.ci_seed_model
"""
import mlflow

from features import clean_data, load_data, split_data
from registry import EXPERIMENT_NAME, register_and_promote, require_tracking_uri
from track_experiments import train_and_log


def main():
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_id, metrics = train_and_log("linreg_baseline", *split_data(clean_data(load_data())))
    version = register_and_promote(run_id)
    print(f"seeded AirbnbPriceModel v{version} @champion (rmse={metrics['rmse']:.2f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rehearse CI locally against a throwaway server** (so the first real CI run isn't the first test)

```bash
# Parentheses: the cd happens only inside the background subshell, so this
# terminal stays in the repo root and the throwaway server's files go to a temp dir.
(cd "$(mktemp -d)" && exec mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5055 > mlflow.log 2>&1) &
until curl -sf http://127.0.0.1:5055/health >/dev/null; do sleep 2; done
MLFLOW_TRACKING_URI=http://127.0.0.1:5055 DATA_PATH=tests/fixtures/listings_sample.csv python -m scripts.ci_seed_model
MLFLOW_TRACKING_URI=http://127.0.0.1:5055 DATA_PATH=tests/fixtures/listings_sample.csv pytest -v
pkill -f "mlflow server.*--port 5055"   # mlflow spawns worker processes; kill them all
lsof -nP -iTCP:5055 -sTCP:LISTEN || echo "port 5055 free"
```
Expected: `seeded AirbnbPriceModel v1 @champion ...` and all tests pass (none skipped).

- [ ] **Step 3: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      MLFLOW_TRACKING_URI: http://127.0.0.1:5000
      # The DVC remote lives on a laptop; CI trains on the committed sample.
      DATA_PATH: tests/fixtures/listings_sample.csv
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Start ephemeral MLflow server
        run: |
          mlflow server --backend-store-uri sqlite:///mlflow.db \
            --artifacts-destination ./mlartifacts \
            --host 127.0.0.1 --port 5000 > mlflow.log 2>&1 &
          for i in $(seq 1 60); do
            curl -sf http://127.0.0.1:5000/health && exit 0
            sleep 2
          done
          cat mlflow.log
          exit 1
      - name: Seed AirbnbPriceModel@champion
        run: python -m scripts.ci_seed_model
      - run: pytest -v

  build-image:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: false
          tags: airbnb-price-api:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
```
(Before committing, check each action's latest major version on its GitHub page and bump if newer.)

- [ ] **Step 4: Commit on `main`, create the GitHub repo, push**

```bash
git add scripts/ci_seed_model.py .github/workflows/ci.yml
git commit -m "ci: test against an ephemeral MLflow server and build the image on PRs"
```
Create an empty repo on GitHub (web UI, or `brew install gh` then `! gh auth login` and `gh repo create`), then:
```bash
git remote add origin git@github.com:<you>/NYC-Airbnb-Price-Prediction.git
git push -u origin main
```

- [ ] **Step 5: Phase 10 DoD — open a real PR and watch CI**

```bash
git switch -c ci-smoke-test
echo "" >> README.md
git commit -am "docs: trigger CI"
git push -u origin ci-smoke-test
```
Open the PR on GitHub. Expected: `CI / test` and `CI / build-image` run and turn green on the PR. Look at the `test` log and confirm `test_model_registry.py` tests **passed** (not skipped). Merge the PR.

---

### Task 12: Deploy workflow → Docker Hub (Phase 10b)

**Files:**
- Create: `.github/workflows/deploy.yml`

**Interfaces:**
- Consumes: repo secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`; input `model_version` (sent by `trigger_deploy`)
- Produces: `<user>/airbnb-price-api:latest`, `:model-v<N>`, `:<git sha>` on Docker Hub.

Note on meaning: the image is model-agnostic (it pulls `@champion` at startup), so `model-v<N>` records *which promotion triggered this release*; it doesn't embed that version.

- [ ] **Step 1: Write `.github/workflows/deploy.yml`**

```yaml
name: Deploy

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: "AirbnbPriceModel version just promoted to @champion"
        required: true
        type: string

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.DOCKERHUB_USERNAME }}/airbnb-price-api:latest
            ${{ secrets.DOCKERHUB_USERNAME }}/airbnb-price-api:model-v${{ inputs.model_version }}
            ${{ secrets.DOCKERHUB_USERNAME }}/airbnb-price-api:${{ github.sha }}
          labels: |
            org.opencontainers.image.revision=${{ github.sha }}
            airbnb.model-version=${{ inputs.model_version }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 2: Add secrets** — Docker Hub → Account settings → Personal access tokens → create a Read & Write token. GitHub repo → Settings → Secrets and variables → Actions → add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

- [ ] **Step 3: Commit via a PR** (CI must stay green), merge to `main` — `workflow_dispatch` only works for workflows on the default branch.

```bash
git switch main && git pull
git switch -c deploy-workflow
git add .github/workflows/deploy.yml
git commit -m "ci: workflow_dispatch deploy that pushes the API image to Docker Hub"
git push -u origin deploy-workflow
```

- [ ] **Step 4: Trigger it the way the Prefect flow will**

Create a fine-grained GitHub PAT for this repo only with **Actions: Read and write**. Then:
```bash
export GITHUB_REPO=<you>/NYC-Airbnb-Price-Prediction
export GITHUB_TOKEN=<pat>
python -m scripts.trigger_deploy --model-version 2
```
Expected: exit code 0; Actions tab shows a `Deploy` run that goes green; Docker Hub shows the three tags.

- [ ] **Step 5: Full loop** — with `GITHUB_REPO`/`GITHUB_TOKEN` exported, run `python orchestrate_training.py`.
Expected: new champion version N, then a `Deploy` run with `model-vN` appears automatically.

- [ ] **Step 6: Pull and run the published image**

```bash
docker run --rm -d --name airbnb-api -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 <you>/airbnb-price-api:latest
sleep 10 && curl -s localhost:8000/health && docker stop airbnb-api
```
Expected: `{"status":"ok",...}`.

---

### Task 13 (optional): Docker Compose — MLflow + API on one network (Phase 11)

**Files:**
- Create: `docker-compose.yml`

**Interfaces:**
- Produces: services `mlflow-server` (port 5000, named volume `mlflow-data`) and `api` (port 8000) with `MLFLOW_TRACKING_URI=http://mlflow-server:5000`.

- [ ] **Step 1: Stop the host MLflow server** (Ctrl-C in its terminal) so port 5000 is free. Confirm the image tag exists: `docker pull ghcr.io/mlflow/mlflow:v$(python -c 'import mlflow; print(mlflow.__version__)')`.

- [ ] **Step 2: Write `docker-compose.yml`** (replace `3.16.1` with your pinned MLflow version; drop `--allowed-hosts` if Task 7 found no such flag)

```yaml
services:
  mlflow-server:
    image: ghcr.io/mlflow/mlflow:v3.16.1
    command: >
      mlflow server
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --artifacts-destination /mlflow/mlartifacts
      --host 0.0.0.0 --port 5000
      --allowed-hosts "localhost:*,127.0.0.1:*,mlflow-server:*"
    volumes:
      - mlflow-data:/mlflow
    ports:
      - "5000:5000"
    healthcheck:
      # The mlflow image has python but not necessarily curl.
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 30s

  api:
    build: .
    environment:
      # The one env var every component reads.
      MLFLOW_TRACKING_URI: http://mlflow-server:5000
    ports:
      - "8000:8000"
    depends_on:
      mlflow-server:
        condition: service_healthy
    restart: on-failure

volumes:
  mlflow-data:
```

- [ ] **Step 3: Bring up MLflow, seed it from the host with the same flow, then start the API**

```bash
docker compose up -d mlflow-server
docker compose ps          # wait until mlflow-server is "healthy"
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python orchestrate_training.py
docker compose up -d api
sleep 10
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"neighbourhood_group":"Manhattan","neighbourhood":"Midtown","latitude":40.7549,"longitude":-73.984,"room_type":"Entire home/apt","minimum_nights":2,"number_of_reviews":20,"reviews_per_month":1.0,"calculated_host_listings_count":1,"availability_365":180}'
```
Expected: `api` only starts after `mlflow-server` is healthy; health OK; a plausible price.

- [ ] **Step 4: Prove healthcheck gating** — `docker compose down && docker compose up -d` then `docker compose logs api | head`: the API starts only after the server reports healthy, and data persists in the `mlflow-data` volume (no reseed needed).

- [ ] **Step 5: Commit via PR**

```bash
git switch -c compose
git add docker-compose.yml
git commit -m "feat: docker compose with healthcheck-gated api and mlflow-server"
git push -u origin compose
```

---

### Task 14: Definition-of-done sweep + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Re-verify every DoD item from spec §5 and tick them in the README**
  - Phase 3: `python train.py` prints RMSE in tens of dollars.
  - Phase 6: 5 runs visible in MLflow UI.
  - Phase 7: separate-process `load_model("models:/AirbnbPriceModel@champion")` predicts.
  - Phase 9: `pytest -v` all green (with server up: none skipped); schema-break experiment done in Task 5.
  - Phase 10: PR shows CI status checks.

- [ ] **Step 2: Complete `README.md`** with: architecture one-paragraph; the setup block from Task 1; "Run it" commands (MLflow server command from Task 7, `python train.py`, `python track_experiments.py`, `python registry.py`, `python orchestrate_training.py`, `uvicorn main:app`, `docker compose up`); the five-run results table from Task 7 Step 8; the champion choice and the RMSE-vs-MAE rule; the cleaning thresholds (drop `$0`, drop `> $800`) and the log-price note; a short "high-cardinality categoricals" note (221 neighbourhoods, one-hot with `handle_unknown="ignore"`); the deviations table from this plan.

- [ ] **Step 3: Commit via PR**

```bash
git switch main && git pull && git switch -c readme
git add README.md
git commit -m "docs: README with results, decisions and run instructions"
git push -u origin readme
```

---

## Self-review

- **Spec coverage:** Phase 1 → T1; Phase 2 → T2; Phase 3 → T3–T4; Phase 4 → T5–T6; Phase 5 → T9; Phase 6 → T7; Phase 7 → T8; Phase 8 → T10; Phase 9 → T3/T5/T6/T7/T8 tests (+ schema-break in T5); Phase 10 → T11–T12; Phase 11 → T13; §5 DoD → T14. Section 2 quirks → `features.py` + tests. Lat/long bounds question → decided in `schemas.py`.
- **Name consistency:** `train_and_log(run_name, X_train, X_test, y_train, y_test) -> (run_id, metrics)` is used identically by `track_experiments.main`, the Prefect task wrapper (`*splits`), `ci_seed_model`, and tests. `register_and_promote(run_id) -> str` and `pick_best(dict[run_id, metrics]) -> run_id` are used identically in `registry.__main__`, the flow, CI seeding and tests. Artifact name `model` ↔ `runs:/<id>/model` everywhere.
- **Known version risks** (flagged inline with checks): MLflow 3 `log_model(name=...)`, `--allowed-hosts` flag, `mlflow-skinny` sufficiency in the image, pandas-3 input-example inference, GitHub Action major versions.
