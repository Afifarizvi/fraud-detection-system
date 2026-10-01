from pydantic import BaseModel, Field
from typing import Optional

class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount")
    ProductCD: int
    card1: int
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: int
    card5: Optional[float] = None
    card6: int
    TransactionDT: int

    class Config:
        extra = "allow"

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
