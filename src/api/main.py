from fastapi import FastAPI, Depends, HTTPException, Header
from src.models.schemas import TransactionRequest, PredictionResponse, HealthResponse
from src.models.predictor import predictor
from src.utils.config import API_KEY, MODEL_VERSION
from src.utils.logger import get_logger
from src.utils.request_logger import log_request, load_recent_requests
from src.utils.drift import compute_drift_report
import time

logger = get_logger("fraud_api")

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost + SHAP fraud detection with explainability",
    version="1.0.0"
)

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": predictor.model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, api_key: str = Depends(verify_api_key)):
    start = time.time()
    payload = transaction.model_dump()

    try:
        result = predictor.predict(payload)
    except Exception as e:
        logger.error("Prediction failed", extra={"extra_data": {"error": str(e)}})
        raise HTTPException(status_code=500, detail="Prediction failed")

    try:
        log_request(payload)
    except Exception as e:
        logger.error("Request logging failed", extra={"extra_data": {"error": str(e)}})

    latency_ms = round((time.time() - start) * 1000, 2)
    logger.info("Prediction made", extra={"extra_data": {
        "fraud_probability": result["fraud_probability"],
        "is_fraud": result["is_fraud"],
        "latency_ms": latency_ms
    }})

    return result

@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionRequest], api_key: str = Depends(verify_api_key)):
    results = []
    for t in transactions:
        payload = t.model_dump()
        results.append(predictor.predict(payload))
        try:
            log_request(payload)
        except Exception:
            pass
    return {"predictions": results, "count": len(results)}

@app.get("/drift")
def drift_report(api_key: str = Depends(verify_api_key)):
    recent_df = load_recent_requests(n=1000)
    if recent_df.empty or len(recent_df) < 30:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 30 logged requests to check drift (have {len(recent_df)})"
        }
    report = compute_drift_report(recent_df)
    return report
