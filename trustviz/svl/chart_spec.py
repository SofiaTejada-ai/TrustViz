from pydantic import BaseModel, Field, conlist, field_validator

class RocSpec(BaseModel):
    title: str = Field(default="ROC Curve", max_length=120)
    fpr: conlist(float, min_length=2)         # v2: min_length
    tpr: conlist(float, min_length=2)         # v2: min_length
    thresholds: conlist(float, min_length=2)  # v2: min_length
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
