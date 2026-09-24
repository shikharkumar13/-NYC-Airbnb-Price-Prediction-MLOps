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
