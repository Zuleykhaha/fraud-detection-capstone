from pydantic import BaseModel
from typing import Any, Dict

class Transaction(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: bool
    used_chip: bool
    used_pin_number: bool
    online_order: bool
    fraud: bool

class Result(BaseModel):
    brf: bool
    iso: bool
    lr: bool
    true: bool
    shap_brf: str
    shap_iso: str
    shap_lr: str

class TestSummary(BaseModel):
    model: str
    accuracy: float = 0.0
    missed_frauds: int = 0
    miss_percentage: float = 0.0
    false_positives: int = 0