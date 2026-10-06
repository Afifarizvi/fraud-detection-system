import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH= "models/reference_distribution.csv"

def compute_drift_report(recent_df: pd.DataFrame, top_n_features: int=15, p_threshold: float=0.05):
    reference_df=pd.read_csv(REFERENCE_PATH)
    common_cols=[c for c in reference_df.columns if c in recent_df.columns]
    results=[]

    for col in common_cols[:top_n_features]:
        ref_vals=reference_df[col].dropna()
        recent_vals=recent_df[col].dropna()

        if len(recent_vals)<30 or len(ref_vals)<30:
            continue

        stat,p_value= ks_2samp(ref_vals, recent_vals)
        is_drifted=p_value<p_threshold

        results.append({
            "feature":col,
            "ks_statistic":round(float(stat),4),
            "p_value":round(float(p_value),6),
            "drifted":bool(is_drifted)
        })

    results.sort(key=lambda r:["ks_statistic"],reverse=True)
    n_drifted=sum(r["drifted"]for r in results)

    return{
        "feature_checked":len(results),
        "feature_drifted":n_drifted,
        "drift_detected":n_drifted>0,
        "sample_size":len(recent_df),
        "details":results
    }
