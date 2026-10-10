from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """A raw transaction. Core fields are typed; any other raw dataset column
    (C1..C14, D1..D15, M1..M9, V1..V339, id_xx) is accepted as-is."""
    model_config = ConfigDict(extra="allow")

    TransactionAmt: float = Field(..., gt=0, description="Transaction amount")
    TransactionDT: int = Field(..., ge=0, description="Seconds since a reference point")
    ProductCD: Optional[str] = None
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float
    model_version: str
    top_contributing_features: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
