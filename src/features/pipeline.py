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

    # Class-level default keeps pipelines pickled before this option existed loadable.
    entity_features = False

    def __init__(self, missing_threshold=0.85, rare_device_min=100, entity_features=False):
        self.missing_threshold = missing_threshold
        self.rare_device_min = rare_device_min
        self.entity_features = entity_features
        self.feature_columns = None

    @staticmethod
    def _col(d, name):
        return d[name] if name in d.columns else pd.Series(np.nan, index=d.index)

    def _uid(self, d):
        """Pseudo-customer id: card1 + addr1 + estimated account start day."""
        day = np.floor(pd.to_numeric(self._col(d, "TransactionDT"), errors="coerce") / 86400)
        d1n = day - pd.to_numeric(self._col(d, "D1"), errors="coerce")
        card1 = pd.to_numeric(self._col(d, "card1"), errors="coerce")
        addr1 = pd.to_numeric(self._col(d, "addr1"), errors="coerce")
        ok = card1.notna() & addr1.notna() & d1n.notna()
        uid = pd.Series(np.nan, index=d.index, dtype=object)
        uid[ok] = (card1[ok].astype(int).astype(str) + "_"
                   + addr1[ok].astype(int).astype(str) + "_"
                   + d1n[ok].astype(int).astype(str))
        return uid

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

        if self.entity_features:
            stats = (pd.DataFrame({"uid": self._uid(raw), "amt": raw["TransactionAmt"]})
                     .dropna(subset=["uid"])
                     .groupby("uid")["amt"].agg(["count", "sum"]))
            self.uid_count = stats["count"]
            self.uid_sum = stats["sum"]

        self.feature_columns = list(self.transform(raw.head(5)).columns)
        return self

    def transform(self, raw: pd.DataFrame, in_sample: bool = False) -> pd.DataFrame:
        """in_sample=True only for the rows the pipeline was fitted on, so a
        training row does not count itself in its own customer history."""
        d = raw.drop(columns=[c for c in (TARGET, ID_COL) if c in raw.columns])
        email_p = self._col(d, "P_emaildomain")
        email_r = self._col(d, "R_emaildomain")

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

        if self.entity_features:
            uid = self._uid(d)
            has = uid.notna()
            n = uid.map(self.uid_count).fillna(0.0)
            s = uid.map(self.uid_sum).fillna(0.0)
            if in_sample:
                n = n - has.astype(float)
                s = s - np.where(has, amt, 0.0)
            known = n > 0
            mean_other = (s / n.where(known, 1.0)).where(known, -999.0)
            out["uid_count_other"] = n
            out["uid_amt_mean_other"] = mean_other
            out["amt_to_uid_mean"] = (amt / mean_other).where(known, -999.0)

        if self.feature_columns:
            out = out[self.feature_columns]
        return out
