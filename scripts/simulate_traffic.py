"""Replay real held-out RAW transactions against the API (true end-to-end test)."""
import argparse

import pandas as pd
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--api-key", required=True)
args = parser.parse_args()

raw = pd.read_csv("data/processed/raw_test_sample.csv")
sample = raw.sample(n=min(args.n, len(raw)), random_state=args.seed)
labels = sample["isFraud"].astype(bool).tolist()
records = [
    {k: v for k, v in r.items() if pd.notna(v)}
    for r in sample.drop(columns=["isFraud", "TransactionID"]).to_dict(orient="records")
]

headers = {"x-api-key": args.api_key}
tp = fp = fn = tn = failed = 0

for payload, actual in zip(records, labels):
    r = requests.post(f"{args.url}/predict", json=payload, headers=headers, timeout=30)
    if r.status_code != 200:
        failed += 1
        continue
    predicted = r.json()["is_fraud"]
    if predicted and actual:
        tp += 1
    elif predicted and not actual:
        fp += 1
    elif actual:
        fn += 1
    else:
        tn += 1

frauds = tp + fn
print(f"Sent {len(records)} raw transactions ({failed} failed)")
print(f"Real frauds: {frauds} | caught: {tp} | missed: {fn} | false alarms: {fp}")
if frauds:
    print(f"Recall: {tp / frauds:.1%}  Precision: {tp / max(tp + fp, 1):.1%}")
