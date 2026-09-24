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
