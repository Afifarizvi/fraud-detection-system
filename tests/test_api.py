import pytest 
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from src.api.main import app
from src.utils.config import API_KEY

client=TestClient(app)

VALID_HEADERS={"x-api-key":API_KEY}

SAMPLE_TRANSACTION={
    "TransactionAmt":150.0,
    "ProductCD":1,
    "card1":5000,
    "card2":200.0,
    "card3":150.0,
    "card4":2,
    "card5":100.0,
    "card6":1,
    "TransactionDT":86400
}
def test_health_check():
    response=client.get("/health")
    assert response.status_code==200
    data=response.json()
    assert data["status"]=="healthy"
    assert data["model_loaded"]is True

def test_predict_without_api_key_returns_401():
    response=client.post("/predict",json=SAMPLE_TRANSACTION)
    assert response.status_code in (401,422)

def test_predict_with_wrong_api_key_returns_401():
    response=client.post(
        "/predict", json=SAMPLE_TRANSACTION, headers={"x-api-key":"wrong-key"}
    )
    assert response.status_code==401

def test_predict_with_valid_data_returns_200():
    response=client.post("/predict",json=SAMPLE_TRANSACTION, headers=VALID_HEADERS)
    assert response.status_code==200
    data=response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <=1.0
    assert isinstance(data["top_contributing_features"],list)
    assert len(data["top_contributing_features"]) >0

def test_predict_response_matches_threshold_logic():
    response=client.post("/predict",json=SAMPLE_TRANSACTION, headers=VALID_HEADERS)
    data=response.json()
    expected_is_fraud=data["fraud_probability"]>=data["threshold_used"]
    assert data["is_fraud"]==expected_is_fraud

def test_predict_rejects_negative_amount():
    bad_transaction=SAMPLE_TRANSACTION.copy()
    bad_transaction["TransactionAmt"]=-50.0
    response=client.post("/predict",json=bad_transaction, headers=VALID_HEADERS)
    assert response.status_code==422

def test_predict_batch_returns_multiple_results():
    batch=[SAMPLE_TRANSACTION, SAMPLE_TRANSACTION]
    response=client.post("/predict/batch",json=batch,headers=VALID_HEADERS)
    assert response.status_code==200
    data=response.json()
    assert data["count"]==2
    assert len(data["predictions"])==2
    
