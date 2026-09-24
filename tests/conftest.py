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
