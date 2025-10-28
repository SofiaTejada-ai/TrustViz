# trustviz/svl/pie_spec.py
from pydantic import BaseModel, field_validator, conlist

class PieSpec(BaseModel):
    title: str = "Pie Chart"
    labels: conlist(str, min_length=1)
    values: conlist(float, min_length=1)
    alt_text: str = "Pie chart showing proportions."

    @field_validator("values")
    @classmethod
    def non_negative(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("Pie values must be ≥ 0.")
        return v

    @field_validator("values")
    @classmethod
    def no_all_zero(cls, v):
        if sum(v) <= 0:
            raise ValueError("Pie values must sum to > 0.")
        return v

    @field_validator("labels")
    @classmethod
    def same_len_labels_values(cls, labels, info):
        values = info.data.get("values")
        if values is not None and len(labels) != len(values):
            raise ValueError("labels and values must have same length.")
        return labels
