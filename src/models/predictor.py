import joblib
import pandas as pd
import shap
from src.utils.config import MODEL_PATH, FEATURE_COLUMNS_PATH, DECISION_THRESHOLD, MODEL_VERSION

class FraudPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        self.explainer = shap.TreeExplainer(self.model)
        self.threshold = DECISION_THRESHOLD
        self.version = MODEL_VERSION

    def _prepare_input(self, payload: dict) -> pd.DataFrame:
        row = {col: payload.get(col, -999) for col in self.feature_columns}
        df = pd.DataFrame([row], columns=self.feature_columns)
        return df

    def predict(self, payload: dict, top_n: int = 5):
        x = self._prepare_input(payload)

        proba = float(self.model.predict_proba(x)[0, 1])
        is_fraud = proba >= self.threshold

        shap_values = self.explainer.shap_values(x)
        contributions = list(zip(self.feature_columns, shap_values[0], x.iloc[0].values))
        contributions.sort(key=lambda c: abs(c[1]), reverse=True)
        top_features = [
            {"feature": f, "shap_value": float(v), "feature_value": float(val)}
            for f, v, val in contributions[:top_n]
        ]

        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(proba, 4),
            "threshold_used": self.threshold,
            "model_version": self.version,
            "top_contributing_features": top_features,
        }

predictor = FraudPredictor()
