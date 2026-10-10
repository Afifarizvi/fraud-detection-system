import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

TARGET = "isFraud"
ID_COL = "TransactionID"


class FeaturePipeline:
    """Raw merged transaction rows -> model-ready features.

    Fitted on training data only, saved with joblib, and reused by the API,
    so training and serving always compute features the same way.
    """

    def __init__(self, missing_threshold=0.85, rare_device_min=100):
        self.missing_threshold = missing_threshold
        self.rare_device_min = rare_device_min
        self.feature_columns = None

    def _clean_cats(self, d, fitting=False):
        cats = pd.DataFrame(index=d.index)
        for c in self.cat_cols:
            col = d[c] if c in d.columns else pd.Series(np.nan, index=d.index)
            cats[c] = col.astype("object").where(col.notna(), "missing").astype(str).astype(object)
        if "DeviceInfo" in cats.columns:
            if fitting:
                counts = cats["DeviceInfo"].value_counts()
                self.device_keep = set(counts[counts >= self.rare_device_min].index)
            cats["DeviceInfo"] = cats["DeviceInfo"].where(
                cats["DeviceInfo"].isin(self.device_keep), "Other"
            )
        return cats

    def fit(self, raw: pd.DataFrame):
        self.feature_columns = None
        raw = raw.drop(columns=[c for c in (TARGET, ID_COL) if c in raw.columns])

        missing = raw.isna().mean()
        self.input_cols = [c for c in raw.columns if missing[c] <= self.missing_threshold]
        self.num_cols = raw[self.input_cols].select_dtypes(include="number").columns.tolist()
        self.cat_cols = [c for c in self.input_cols if c not in self.num_cols]

        cats = self._clean_cats(raw, fitting=True)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder.fit(cats)

        self.card1_count = raw.groupby("card1")["TransactionAmt"].count()
        self.card1_mean = raw.groupby("card1")["TransactionAmt"].mean()
        self.count_fallback = float(self.card1_count.median())
        self.mean_fallback = float(self.card1_mean.median())

        self.feature_columns = list(self.transform(raw.head(5)).columns)
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        d = raw.drop(columns=[c for c in (TARGET, ID_COL) if c in raw.columns])
        email_p = d["P_emaildomain"] if "P_emaildomain" in d.columns else pd.Series(np.nan, index=d.index)
        email_r = d["R_emaildomain"] if "R_emaildomain" in d.columns else pd.Series(np.nan, index=d.index)

        base = d.reindex(columns=self.input_cols)
        num = base[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(-999)
        cats = self._clean_cats(base)
        cats_enc = pd.DataFrame(self.encoder.transform(cats), columns=self.cat_cols, index=base.index)
        out = pd.concat([num, cats_enc], axis=1)

        dt, amt = out["TransactionDT"], out["TransactionAmt"]
        out["hour"] = (dt // 3600) % 24
        out["day_of_week"] = (dt // 86400) % 7
        out["amt_is_round"] = (amt % 1 == 0).astype(int)
        out["amt_cents"] = amt % 1
        out["amt_log"] = np.log1p(amt)

        card1 = base["card1"]
        out["card1_count"] = card1.map(self.card1_count).fillna(self.count_fallback)
        out["card1_amt_mean"] = card1.map(self.card1_mean).fillna(self.mean_fallback)
        out["amt_to_card1_mean"] = amt / out["card1_amt_mean"]

        both = email_p.notna() & email_r.notna()
        out["email_match"] = np.where(both, (email_p == email_r).astype(int), -1)

        if self.feature_columns:
            out = out[self.feature_columns]
        return out
