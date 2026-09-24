from pathlib import Path

import pytest

from features import clean_data, load_data, split_data

SAMPLE_PATH = Path(__file__).parent / "fixtures" / "listings_sample.csv"


@pytest.fixture(scope="session")
def sample_splits():
    """(X_train, X_test, y_train, y_test) from the committed 2,000-row sample."""
    return split_data(clean_data(load_data(SAMPLE_PATH)))
