"""Train the fraud model from raw data and save every artifact the API needs.

Run from the project root:
  python -m src.train --features compare             # base vs entity features, winner picked on validation
  python -m src.train --features entity --tune 20    # re-tune hyperparameters on validation data
  python -m src.train --features entity --miss-cost 20
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.features.pipeline import FeaturePipeline, TARGET, ID_COL

RAW_DIR = "data/raw"
MODEL_DIR = "models"
DEFAULT_PARAMS = dict(
    n_estimators=272, max_depth=7, learning_rate=0.0963,
    subsample=0.7803, colsample_bytree=0.9988, min_child_weight=6,
    random_state=42, n_jobs=-1,
)
COST_RATIOS = [1.0, 5.0, 10.0, 20.0]


def load_split():
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
    return train, val, test


def recall_at_fpr(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target, fpr, tpr))


def pos_weight(y):
    return float((y == 0).sum() / (y == 1).sum())


def tune(X_train, y_train, X_val, y_val, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spw = pos_weight(y_train)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 500),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            random_state=42, n_jobs=-1,
        )
        m = xgb.XGBClassifier(**params, scale_pos_weight=spw).fit(X_train, y_train)
        return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print(f"best validation PR-AUC from tuning: {study.best_value:.4f}")
    return {**study.best_params, "random_state": 42, "n_jobs": -1}


def run_variant(name, entity, train, val, n_tune):
    print(f"\n--- variant: {name} ---")
    pipe = FeaturePipeline(entity_features=entity).fit(train)
    X_train = pipe.transform(train, in_sample=True)
    X_val = pipe.transform(val)
    y_train, y_val = train[TARGET].to_numpy(), val[TARGET].to_numpy()
    print(f"features: {X_train.shape[1]}")

    params = tune(X_train, y_train, X_val, y_val, n_tune) if n_tune else dict(DEFAULT_PARAMS)
    model = xgb.XGBClassifier(**params, scale_pos_weight=pos_weight(y_train))
    model.fit(X_train, y_train)

    p_val = model.predict_proba(X_val)[:, 1]
    scores = {
        "pr_auc": float(average_precision_score(y_val, p_val)),
        "auc": float(roc_auc_score(y_val, p_val)),
        "recall_at_1pct_fpr": recall_at_fpr(y_val, p_val, 0.01),
    }
    return {
        "name": name, "pipe": pipe, "model": model, "params": params, "p_val": p_val,
        "val_scores": scores, "n_features": int(X_train.shape[1]),
        "reference": X_train.sample(5000, random_state=42),
    }


def pick_threshold(y, p, miss_cost):
    """Threshold minimising (miss_cost * missed frauds + false alarms)."""
    best_t, best_cost = 0.5, None
    for t in np.arange(0.05, 0.99, 0.01):
        pred = p >= t
        cost = miss_cost * int((~pred & (y == 1)).sum()) + int((pred & (y == 0)).sum())
        if best_cost is None or cost < best_cost:
            best_t, best_cost = float(t), cost
    return round(best_t, 2)


def operating_point(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {
        "threshold": t, "caught": tp, "missed": fn, "false_alarms": fp,
        "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
        "false_alarm_rate": fp / max(int((y == 0).sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["base", "entity", "compare"], default="compare")
    ap.add_argument("--tune", type=int, default=0, help="Optuna trials on validation PR-AUC")
    ap.add_argument("--miss-cost", type=float, default=10.0,
                    help="how many false alarms one missed fraud is worth")
    ap.add_argument("--version", default="v3")
    args = ap.parse_args()
    if args.features == "compare" and args.tune:
        ap.error("--tune works on one variant: use --features base or --features entity")

    train, val, test = load_split()
    flags = {"base": False, "entity": True}
    names = ["base", "entity"] if args.features == "compare" else [args.features]
    results = [run_variant(n, flags[n], train, val, args.tune) for n in names]

    print("\n=== Validation comparison (nothing is tuned on the test set) ===")
    print(f"{'variant':<8}{'features':>9}{'pr_auc':>9}{'auc':>9}{'recall@1%FPR':>14}")
    for r in results:
        s = r["val_scores"]
        print(f"{r['name']:<8}{r['n_features']:>9}{s['pr_auc']:>9.4f}{s['auc']:>9.4f}{s['recall_at_1pct_fpr']:>14.4f}")
    winner = max(results, key=lambda r: r["val_scores"]["pr_auc"])
    print(f"winner by validation PR-AUC: {winner['name']}")

    pipe, model = winner["pipe"], winner["model"]
    y_val, y_test = val[TARGET].to_numpy(), test[TARGET].to_numpy()
    p_test = model.predict_proba(pipe.transform(test))[:, 1]

    ratios = sorted({float(x) for x in COST_RATIOS} | {float(args.miss_cost)})
    ops = {}
    for r in ratios:
        t = pick_threshold(y_val, winner["p_val"], r)
        ops[str(r)] = operating_point(y_test, p_test, t)
    threshold = ops[str(float(args.miss_cost))]["threshold"]

    test_scores = {
        "auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "recall_at_1pct_fpr": recall_at_fpr(y_test, p_test, 0.01),
        "recall_at_5pct_fpr": recall_at_fpr(y_test, p_test, 0.05),
    }

    print(f"\n=== Test results for '{winner['name']}' ===")
    for k, v in test_scores.items():
        print(f"{k:>20}: {v:.4f}")
    print("\nOperating points (thresholds chosen on validation, applied to test)")
    print(f"{'miss cost':>9}{'thresh':>8}{'caught':>8}{'missed':>8}{'false alarms':>14}{'recall':>8}{'precision':>10}")
    for r in ratios:
        o = ops[str(r)]
        print(f"{r:>9g}{o['threshold']:>8.2f}{o['caught']:>8}{o['missed']:>8}{o['false_alarms']:>14}"
              f"{o['recall']:>8.3f}{o['precision']:>10.3f}")
    print(f"\nSaved threshold (miss cost {args.miss_cost:g}): {threshold}")

    v = args.version
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(pipe, f"{MODEL_DIR}/pipeline_{v}.pkl")
    joblib.dump(model, f"{MODEL_DIR}/xgb_fraud_model_{v}.pkl")
    winner["reference"].to_csv(f"{MODEL_DIR}/reference_{v}.csv.gz", index=False)
    metrics = {
        "version": v, "features": winner["name"], "n_features": winner["n_features"],
        "threshold": threshold, "miss_cost": args.miss_cost,
        "validation": {r["name"]: r["val_scores"] for r in results},
        "test": test_scores, "operating_points": ops, "params": winner["params"],
    }
    with open(f"{MODEL_DIR}/metrics_{v}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    sample_path = "data/processed/raw_test_sample.csv"
    if not os.path.exists(sample_path):
        test.sample(2000, random_state=42).to_csv(sample_path, index=False)

    try:
        import mlflow
        mlflow.set_experiment("fraud-detection-baseline")
        with mlflow.start_run(run_name=f"{v}_{winner['name']}_features"):
            mlflow.log_params({k: val_ for k, val_ in winner["params"].items()})
            mlflow.log_param("features", winner["name"])
            mlflow.log_param("miss_cost", args.miss_cost)
            mlflow.log_metrics(test_scores)
    except Exception as e:
        print("MLflow logging skipped:", e)


if __name__ == "__main__":
    main()
