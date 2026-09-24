from pathlib import Path

import mlflow
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
