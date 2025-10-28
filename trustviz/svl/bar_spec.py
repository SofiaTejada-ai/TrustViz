from pydantic import BaseModel, Field, conlist, field_validator, model_validator

class BarSpec(BaseModel):
    title: str = Field(default="Bar Chart", max_length=120)
    x: conlist(str, min_length=1)
    y: conlist(float, min_length=1)
    x_label: str = "Category"
    y_label: str = "Value"
    alt_text: str = Field(
        default="Bar chart showing values by category.",
        min_length=10, max_length=240
    )

    @field_validator("x")
    @classmethod
    def no_empty_labels(cls, v):
        if any(not s.strip() for s in v):
            raise ValueError("Category labels must be non-empty.")
        return v

    @model_validator(mode="after")
    def check_lengths(self):
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length.")
        if len(self.x) > 30:
            raise ValueError("Too many categories; cap at 30.")
        return self
