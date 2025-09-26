from typing import List, Optional
from pydantic import BaseModel, Field, conlist, field_validator

class RocSpec(BaseModel):
    title: str = Field(default="ROC Curve", max_length=120)
    fpr: conlist(float, min_length=2)
    tpr: conlist(float, min_length=2)
    # Allow None (null in JSON) for thresholds entries like +inf
    thresholds: List[Optional[float]]
    auc: float
    x_label: str = "False Positive Rate"
    y_label: str = "True Positive Rate"
    alt_text: str = Field(
        default="ROC curve showing model trade-off.",
        min_length=10, max_length=240
    )

    @field_validator("fpr", "tpr")
    @classmethod
    def values_in_unit_interval(cls, v):
        if not all(0.0 <= x <= 1.0 for x in v):
            raise ValueError("ROC values must be within [0,1].")
        return v

    @field_validator("thresholds")
    @classmethod
    def thresholds_are_finite_or_null(cls, v):
        # Accept None, or finite floats
        from math import isfinite
        if not all((x is None) or (isinstance(x, (int, float)) and isfinite(float(x))) for x in v):
            raise ValueError("thresholds must be finite numbers or null.")
        if len(v) < 2:
            raise ValueError("thresholds must have at least 2 entries.")
        return v
