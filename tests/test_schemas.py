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
