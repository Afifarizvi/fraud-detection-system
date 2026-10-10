"""Train the fraud model from raw data and save every artifact the API needs.

Run from the project root:  python -m src.train
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.features.pipeline import FeaturePipeline, TARGET, ID_COL

RAW_DIR = "data/raw"
MODEL_DIR = "models"

# Best parameters from the earlier 20-trial Optuna search
PARAMS = dict(
    n_estimators=272, max_depth=7, learning_rate=0.0963,
    subsample=0.7803, colsample_bytree=0.9988, min_child_weight=6,
    random_state=42, eval_metric="auc", n_jobs=-1,
)


def main():
    print("Loading raw data...")
    tx = pd.read_csv(f"{RAW_DIR}/train_transaction.csv")
    idn = pd.read_csv(f"{RAW_DIR}/train_identity.csv")
    df = tx.merge(idn, on=ID_COL, how="left")
    del tx, idn
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n = len(df)
    i_val, i_test = int(n * 0.72), int(n * 0.80)
    train, val, test = df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print("Fitting feature pipeline on train only...")
    pipe = FeaturePipeline().fit(train)
    X_train, y_train = pipe.transform(train), train[TARGET]
    X_val, y_val = pipe.transform(val), val[TARGET]
    X_test, y_test = pipe.transform(test), test[TARGET]
    print(f"features: {X_train.shape[1]}")

    print("Training...")
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(**PARAMS, scale_pos_weight=spw)
    model.fit(X_train, y_train)

    # Choose the threshold on validation data, never on the test set
    p_val = model.predict_proba(X_val)[:, 1]
    grid = np.arange(0.30, 0.96, 0.01)
    f1s = [f1_score(y_val, (p_val >= t).astype(int)) for t in grid]
    threshold = float(grid[int(np.argmax(f1s))])

    p_test = model.predict_proba(X_test)[:, 1]
    pred = (p_test >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, p_test)
    metrics = {
        "threshold": round(threshold, 2),
        "auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "recall_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "recall_at_5pct_fpr": float(np.interp(0.05, fpr, tpr)),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(pipe, f"{MODEL_DIR}/pipeline_v2.pkl")
    joblib.dump(model, f"{MODEL_DIR}/xgb_fraud_model_v2.pkl")
    X_train.sample(5000, random_state=42).to_csv(f"{MODEL_DIR}/reference_distribution.csv", index=False)
    with open(f"{MODEL_DIR}/metrics_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # Raw held-out rows for end-to-end API tests and traffic replay
    test.sample(2000, random_state=42).to_csv("data/processed/raw_test_sample.csv", index=False)

    try:
        import mlflow
        mlflow.set_experiment("fraud-detection-baseline")
        with mlflow.start_run(run_name="v2_pipeline_from_raw"):
            mlflow.log_params({k: v for k, v in PARAMS.items() if k != "eval_metric"})
            mlflow.log_param("split", "time-based 72/8/20, threshold tuned on val")
            mlflow.log_metrics(metrics)
    except Exception as e:
        print("MLflow logging skipped:", e)

    print("\n=== Held-out test results ===")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.4f}")


if __name__ == "__main__":
    main()
