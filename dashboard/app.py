import os

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Fraud Detection Monitor", layout="wide")

SAMPLE_PATH = "data/processed/raw_test_sample.csv"

# ---------------- sidebar ----------------
st.sidebar.title("Fraud Detection Monitor")
api_url = st.sidebar.text_input(
    "API URL", os.getenv("DASHBOARD_API_URL", "http://127.0.0.1:8000")
).rstrip("/")
api_key = os.getenv("DASHBOARD_API_KEY", "")
st.sidebar.button("Refresh")


def call(method, path, **kwargs):
    r = requests.request(method, f"{api_url}{path}", headers={"x-api-key": api_key},
                         timeout=60, **kwargs)
    r.raise_for_status()
    return r.json()


def safe_get(path):
    try:
        return call("GET", path)
    except Exception as e:
        st.error(f"Request to {path} failed: {e}")
        return None


@st.cache_data
def load_sample():
    if os.path.exists(SAMPLE_PATH):
        return pd.read_csv(SAMPLE_PATH)
    return None


def shap_chart(contribs):
    d = pd.DataFrame(contribs)
    d["label"] = d.apply(lambda r: f"{r['feature']} = {r['feature_value']:g}", axis=1)
    d["direction"] = np.where(d["shap_value"] > 0, "toward fraud", "toward legitimate")
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("shap_value:Q", title="Impact on fraud score (SHAP)"),
        y=alt.Y("label:N", sort=list(d["label"]), title=None),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["toward fraud", "toward legitimate"],
                            range=["#d62728", "#1f77b4"]),
            legend=alt.Legend(title=None),
        ),
        tooltip=["feature", "shap_value", "feature_value"],
    ).properties(height=260, width="container")


try:
    health = call("GET", "/health")
except Exception as e:
    st.error(f"Cannot reach the API at {api_url}: {e}")
    st.stop()
st.sidebar.success(f"API healthy - model {health['model_version']}")

tab1, tab2, tab3 = st.tabs(["Live monitoring", "Drift", "Score a transaction"])

# ---------------- tab 1: live monitoring ----------------
with tab1:
    stats = safe_get("/stats")
    if stats is not None:
        if stats.get("total_predictions", 0) == 0:
            st.info("No predictions logged yet. Run scripts/simulate_traffic.py or score a transaction.")
        else:
            threshold = stats.get("threshold")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Predictions", f"{stats['total_predictions']:,}")
            m2.metric("Flagged as fraud", f"{stats['fraud_flagged']:,}")
            m3.metric("Flag rate", f"{stats['fraud_rate']:.1%}")
            m4.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
            m5.metric("p95 latency", f"{stats['p95_latency_ms']:.0f} ms")

            recent = pd.DataFrame(stats["recent"])
            recent["timestamp"] = pd.to_datetime(recent["timestamp"], format="ISO8601")
            recent["decision"] = np.where(recent["is_fraud"] == 1, "flagged", "not flagged")

            points = alt.Chart(recent).mark_circle(size=45, opacity=0.7).encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("fraud_probability:Q", title="Fraud probability",
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "decision:N",
                    scale=alt.Scale(domain=["flagged", "not flagged"], range=["#d62728", "#1f77b4"]),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["timestamp:T", "fraud_probability:Q", "latency_ms:Q"],
            )
            layers = [points]
            if threshold is not None:
                layers.append(
                    alt.Chart(pd.DataFrame({"y": [threshold]}))
                    .mark_rule(strokeDash=[5, 4], color="gray").encode(y="y:Q")
                )
            st.subheader(f"Recent scores (last {len(recent)} requests)")
            st.altair_chart(alt.layer(*layers).properties(height=300, width="container"))
            if threshold is not None:
                st.caption(f"Dashed line = decision threshold ({threshold}).")

            left, right = st.columns(2)
            with left:
                st.subheader("Score distribution")
                counts, edges = np.histogram(recent["fraud_probability"], bins=20, range=(0, 1))
                st.bar_chart(pd.DataFrame({"count": counts}, index=[f"{e:.2f}" for e in edges[:-1]]))
            with right:
                st.subheader("Flag rate (rolling 50 requests)")
                rolling = recent["is_fraud"].rolling(50, min_periods=50).mean()
                st.line_chart(rolling.reset_index(drop=True))

# ---------------- tab 2: drift ----------------
with tab2:
    drift = safe_get("/drift")
    if drift is not None:
        if drift.get("status") == "insufficient_data":
            st.info(drift["message"])
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("Features checked", drift["features_checked"])
            k2.metric("Features drifted", drift["features_drifted"])
            k3.metric("Requests compared", f"{drift['sample_size']:,}")
            if drift["drift_detected"]:
                st.error("Drift detected: live traffic differs from the training data on some features.")
            else:
                st.success(f"No feature crossed the corrected significance cutoff (alpha = {drift['alpha']}).")
            det = pd.DataFrame(drift["details"])
            st.subheader("KS statistic per feature (higher = bigger shift)")
            st.bar_chart(det.set_index("feature")["ks_statistic"])
            st.dataframe(det, hide_index=True)
            st.caption(
                "Each feature's live distribution is compared with a sample of the training data using a "
                "Kolmogorov-Smirnov test. The significance cutoff is Bonferroni-corrected for the number "
                "of features tested, so a handful of near-misses does not trigger false alarms."
            )

# ---------------- tab 3: score a transaction ----------------
with tab3:
    st.write("Send one raw transaction through the live API and see why the model scored it this way. "
             "Scored transactions are also logged, so they show up in the monitoring tabs.")
    sample = load_sample()
    modes = ["Enter manually"]
    if sample is not None:
        modes += ["Real held-out fraud", "Real held-out legit"]
    mode = st.radio("Input", modes, horizontal=True)
    payload, actual = None, None

    if mode == "Enter manually":
        with st.form("manual"):
            r1a, r1b, r1c = st.columns(3)
            amount = r1a.number_input("TransactionAmt", min_value=0.01, value=150.0, step=1.0)
            product = r1b.selectbox("ProductCD", ["W", "C", "H", "R", "S"])
            dt = r1c.number_input("TransactionDT (seconds)", min_value=0, value=86400 * 150, step=3600)
            r2a, r2b, r2c = st.columns(3)
            card1 = r2a.number_input("card1", min_value=0, value=13926, step=1)
            card4 = r2b.selectbox("card4", ["visa", "mastercard", "discover", "american express"])
            card6 = r2c.selectbox("card6", ["debit", "credit"])
            r3a, r3b, r3c = st.columns(3)
            p_email = r3a.text_input("P_emaildomain", "gmail.com")
            r_email = r3b.text_input("R_emaildomain (optional)", "")
            addr1 = r3c.number_input("addr1", min_value=0, value=315, step=1)
            submitted = st.form_submit_button("Score transaction")
        if submitted:
            payload = {"TransactionAmt": float(amount), "TransactionDT": int(dt),
                       "ProductCD": product, "card1": int(card1), "card4": card4,
                       "card6": card6, "addr1": float(addr1)}
            if p_email:
                payload["P_emaildomain"] = p_email
            if r_email:
                payload["R_emaildomain"] = r_email
    elif st.button("Pick a random one and score it"):
        want = 1 if mode.endswith("fraud") else 0
        row = sample[sample["isFraud"] == want].sample(1).iloc[0]
        actual = "FRAUD" if want == 1 else "legitimate"
        payload = {k: (v.item() if hasattr(v, "item") else v)
                   for k, v in row.drop(["isFraud", "TransactionID"]).items() if pd.notna(v)}

    if payload is not None:
        try:
            result = call("POST", "/predict", json=payload)
        except Exception as e:
            st.error(f"Scoring failed: {e}")
        else:
            left, right = st.columns([1, 2])
            with left:
                st.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
                st.caption(f"Decision threshold: {result['threshold_used']} - model {result['model_version']}")
                if result["is_fraud"]:
                    st.error("Flagged as FRAUD")
                else:
                    st.success("Looks legitimate")
                if actual:
                    st.caption(f"Actual label in the held-out data: {actual}")
            with right:
                st.subheader("Why? Top feature contributions")
                st.altair_chart(shap_chart(result["top_contributing_features"]))
                st.caption("Red pushes the score toward fraud, blue toward legitimate. The value after each "
                           "feature name is the (encoded) input the model actually saw.")
