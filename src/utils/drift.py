import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH = "models/reference_distribution.csv"


def compute_drift_report(recent_df: pd.DataFrame, features: list, min_samples: int = 30):
    reference_df = pd.read_csv(REFERENCE_PATH)
    alpha = 0.05 / max(len(features), 1)
    results = []

    for col in features:
        if col not in reference_df.columns or col not in recent_df.columns:
            continue

        ref_vals = reference_df[col].dropna()
        recent_vals = recent_df[col].dropna()
        if len(ref_vals) < min_samples or len(recent_vals) < min_samples:
            continue

        stat, p_value = ks_2samp(ref_vals, recent_vals)
        results.append({
            "feature": col,
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "drifted": bool(p_value < alpha),
        })

    results.sort(key=lambda r: r["ks_statistic"], reverse=True)
    n_drifted = sum(r["drifted"] for r in results)

    return {
        "features_checked": len(results),
        "features_drifted": n_drifted,
        "drift_detected": n_drifted > 0,
        "alpha": round(alpha, 5),
        "sample_size": int(len(recent_df)),
        "details": results,
    }