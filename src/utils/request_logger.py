import os
import threading

import pandas as pd

LOG_PATH = "logs/prediction_requests.csv"
_lock = threading.Lock()


def log_request(features: pd.DataFrame):
    """Log the model-ready feature row, so drift compares like with like."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with _lock:
        write_header = not os.path.exists(LOG_PATH)
        features.to_csv(LOG_PATH, mode="a", header=write_header, index=False)


def load_recent_requests(n: int = 1000) -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    return pd.read_csv(LOG_PATH).tail(n)
