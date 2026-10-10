"""Score the same raw rows offline and compare with what the live API returned."""
import json

import joblib
import pandas as pd

pipe = joblib.load("models/pipeline_v2.pkl")
model = joblib.load("models/xgb_fraud_model_v2.pkl")
with open("models/metrics_v2.json") as f:
    threshold = json.load(f)["threshold"]

raw = pd.read_csv("data/processed/raw_test_sample.csv")
y = raw["isFraud"].astype(bool)
proba = model.predict_proba(pipe.transform(raw))[:, 1]
pred = proba >= threshold

tp = int((pred & y).sum())
fn = int((~pred & y).sum())
fp = int((pred & ~y).sum())
print(f"Offline on the same {len(raw)} rows (threshold {threshold}):")
print(f"Real frauds: {tp + fn} | caught: {tp} | missed: {fn} | false alarms: {fp}")
print(f"Recall: {tp / (tp + fn):.1%}  Precision: {tp / max(tp + fp, 1):.1%}")
