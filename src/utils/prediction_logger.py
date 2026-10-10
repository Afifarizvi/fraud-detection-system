import os
import threading
from datetime import datetime, timezone
import pandas as pd

PRED_LOG_PATH="logs/predictions.csv"
COLUMNS=["timestamp","fraud_probability","is_fraud","latency_ms"]
_lock=threading.Lock()

def log_prediction(fraud_probability:float, is_fraud:bool, latency_ms:float):
    os.makedirs(os.path.dirname(PRED_LOG_PATH), exist_ok=True)
    row=pd.DataFrame([{
        "timestamp":datetime.now(timezone.utc).isoformat(),
        "fraud_probability": fraud_probability,
        "is_fraud":int(is_fraud),
        "latency_ms":latency_ms,
    }], columns=COLUMNS)
    with _lock:
        write_header=not os.path.exists(PRED_LOG_PATH)
        row.to_csv(PRED_LOG_PATH, mode="a", header=write_header, index=False)

def load_predictions() -> pd.DataFrame:
    if not os.path.exists(PRED_LOG_PATH):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(PRED_LOG_PATH)

