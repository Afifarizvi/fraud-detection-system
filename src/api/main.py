import time

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException

from src.models.predictor import predictor
from src.models.schemas import HealthResponse, PredictionResponse, TransactionRequest
from src.utils.config import API_KEY, MODEL_VERSION
from src.utils.drift import compute_drift_report
from src.utils.logger import get_logger
from src.utils.prediction_logger import load_predictions, log_prediction
from src.utils.request_logger import load_recent_requests, log_request

logger = get_logger("fraud_api")

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost + SHAP fraud detection with explainability",
    version="2.0.0",
)

# Monitor drift on the model's most important features.
# TransactionDT is skipped: it is a raw timestamp, so it always "drifts".
_importances = predictor.model.feature_importances_
TOP_FEATURES = [
    predictor.feature_columns[i]
    for i in np.argsort(_importances)[::-1]
    if predictor.feature_columns[i] != "TransactionDT"
][:20]


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def score_and_log(payload: dict):
    start = time.time()
    x = predictor.featurize(payload)
    result = predictor.score(x)
    latency_ms = round((time.time() - start) * 1000, 2)
    try:
        log_request(x)
        log_prediction(result["fraud_probability"], result["is_fraud"], latency_ms)
    except Exception as e:
        logger.error("Logging failed", extra={"extra_data": {"error": str(e)}})
    return result, latency_ms


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": predictor.model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, api_key: str = Depends(verify_api_key)):
    payload = transaction.model_dump(exclude_none=True)
    try:
        result, latency_ms = score_and_log(payload)
    except Exception as e:
        logger.error("Prediction failed", extra={"extra_data": {"error": str(e)}})
        raise HTTPException(status_code=500, detail="Prediction failed")

    logger.info("Prediction made", extra={"extra_data": {
        "fraud_probability": result["fraud_probability"],
        "is_fraud": result["is_fraud"],
        "latency_ms": latency_ms,
    }})
    return result


@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionRequest], api_key: str = Depends(verify_api_key)):
    results = []
    for t in transactions:
        result, _ = score_and_log(t.model_dump(exclude_none=True))
        results.append(result)
    return {"predictions": results, "count": len(results)}


@app.get("/stats")
def stats(api_key: str = Depends(verify_api_key)):
    df = load_predictions()
    if df.empty:
        return {"total_predictions": 0}
    return {
        "total_predictions": int(len(df)),
        "fraud_flagged": int(df["is_fraud"].sum()),
        "fraud_rate": round(float(df["is_fraud"].mean()), 4),
        "avg_latency_ms": round(float(df["latency_ms"].mean()), 2),
        "p95_latency_ms": round(float(df["latency_ms"].quantile(0.95)), 2),
        "recent": df.tail(500).to_dict(orient="records"),
    }


@app.get("/drift")
def drift_report(api_key: str = Depends(verify_api_key)):
    recent_df = load_recent_requests(n=1000)
    if len(recent_df) < 30:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 30 logged requests to check drift (have {len(recent_df)})",
            "logged_requests": int(len(recent_df)),
        }
    return compute_drift_report(recent_df, TOP_FEATURES)
