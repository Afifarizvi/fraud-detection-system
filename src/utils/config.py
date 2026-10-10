import json
import os

from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_fraud_model_v2.pkl")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "models/pipeline_v2.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "models/metrics_v2.json")


def _threshold_from_training() -> float:
    try:
        with open(METRICS_PATH) as f:
            return float(json.load(f)["threshold"])
    except (OSError, KeyError, ValueError):
        return 0.75


DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", _threshold_from_training()))
API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")
