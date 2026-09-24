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
            "MLFLOW_TRACKING_URI is not set. Start the server (see implementation.md, Task 7) and run:\n"
            "  export MLFLOW_TRACKING_URI=http://127.0.0.1:5001"
        )
    return uri
