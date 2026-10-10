import joblib
import pandas as pd
import shap

from src.utils.config import (DECISION_THRESHOLD, MODEL_PATH, MODEL_VERSION,
                              PIPELINE_PATH)


class FraudPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.pipeline = joblib.load(PIPELINE_PATH)
        self.feature_columns = self.pipeline.feature_columns
        self.explainer = shap.TreeExplainer(self.model)
        self.threshold = DECISION_THRESHOLD
        self.version = MODEL_VERSION

    def featurize(self, payload: dict) -> pd.DataFrame:
        """Raw transaction -> model-ready one-row frame (same code as training)."""
        return self.pipeline.transform(pd.DataFrame([payload]))

    def score(self, x: pd.DataFrame, top_n: int = 5) -> dict:
        proba = float(self.model.predict_proba(x)[0, 1])
        shap_values = self.explainer.shap_values(x)
        contributions = sorted(
            zip(self.feature_columns, shap_values[0], x.iloc[0].values),
            key=lambda c: abs(c[1]),
            reverse=True,
        )
        return {
            "is_fraud": bool(proba >= self.threshold),
            "fraud_probability": round(proba, 4),
            "threshold_used": self.threshold,
            "model_version": self.version,
            "top_contributing_features": [
                {"feature": f, "shap_value": float(v), "feature_value": float(val)}
                for f, v, val in contributions[:top_n]
            ],
        }

    def predict(self, payload: dict, top_n: int = 5) -> dict:
        return self.score(self.featurize(payload), top_n)


predictor = FraudPredictor()
