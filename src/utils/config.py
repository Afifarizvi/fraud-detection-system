import os
from dotenv import load_dotenv
load_dotenv()
MODEL_PATH=os.getenv("MODEL_PATH","models/xgb_fraud_model_final.pkl")
FEATURE_COLUMNS_PATH=os.getenv("FEATURE_COLUMN_PATH","models/feature_columns_final.pkl")
DECISION_THRESHOLD=float(os.getenv("DECISION_THRESHOLD","0.8"))
API_KEY=os.getenv("API_KEY","dev-key-change-in-production")
MODEL_VERSION=os.getenv("MODEL_VERSION","v1-tuned")
