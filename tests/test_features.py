import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import features


@pytest.fixture
def raw_df():
    """Six raw listings with all 16 CSV columns and the dataset's real quirks (implementation.md, Task 2)."""
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
