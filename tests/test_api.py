import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app
from src.models.predictor import predictor
from src.utils.config import API_KEY

client = TestClient(app)
HEADERS = {"x-api-key": API_KEY}

SAMPLE = {
    "TransactionAmt": 150.0,
    "TransactionDT": 86400,
    "ProductCD": "W",
    "card1": 5000,
    "card2": 200.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "credit",
    "addr1": 315.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
}


@pytest.fixture(autouse=True)
def isolate_logs(tmp_path, monkeypatch):
    import src.utils.prediction_logger as pl
    import src.utils.request_logger as rl
    monkeypatch.setattr(rl, "LOG_PATH", str(tmp_path / "requests.csv"))
    monkeypatch.setattr(pl, "PRED_LOG_PATH", str(tmp_path / "predictions.csv"))


def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["model_loaded"] is True


def test_predict_without_api_key_is_rejected():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code in (401, 422)


def test_predict_with_wrong_api_key_returns_401():
    r = client.post("/predict", json=SAMPLE, headers={"x-api-key": "wrong-key"})
    assert r.status_code == 401


def test_predict_with_valid_data_returns_200():
    r = client.post("/predict", json=SAMPLE, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert len(data["top_contributing_features"]) > 0


def test_is_fraud_follows_threshold():
    data = client.post("/predict", json=SAMPLE, headers=HEADERS).json()
    assert data["is_fraud"] == (data["fraud_probability"] >= data["threshold_used"])


def test_rejects_negative_amount():
    r = client.post("/predict", json={**SAMPLE, "TransactionAmt": -50.0}, headers=HEADERS)
    assert r.status_code == 422


def test_batch_returns_multiple_results():
    r = client.post("/predict/batch", json=[SAMPLE, SAMPLE], headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_engineered_features_are_computed_server_side():
    x = predictor.featurize(SAMPLE)
    assert list(x.columns) == predictor.feature_columns
    assert x["hour"].iloc[0] == 0
    assert x["day_of_week"].iloc[0] == 1
    assert x["email_match"].iloc[0] == 1
    assert x["card1_amt_mean"].iloc[0] != -999
    assert x["amt_log"].iloc[0] == pytest.approx(np.log1p(150.0))


def test_unseen_category_does_not_crash():
    r = client.post("/predict", json={**SAMPLE, "ProductCD": "NEVER_SEEN"}, headers=HEADERS)
    assert r.status_code == 200


def test_single_row_features_match_batch_features():
    rows = [
        SAMPLE,
        {**SAMPLE, "TransactionAmt": 9.99, "card1": 1234, "card4": "mastercard"},
        {"TransactionAmt": 20.0, "TransactionDT": 5000},
    ]
    batch = predictor.pipeline.transform(pd.DataFrame(rows))
    for i, row in enumerate(rows):
        single = predictor.featurize(row)
        pd.testing.assert_frame_equal(
            batch.iloc[[i]].reset_index(drop=True),
            single.reset_index(drop=True),
            check_dtype=False,
        )
