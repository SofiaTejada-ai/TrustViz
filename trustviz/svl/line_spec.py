# trustviz/svl/line_spec.py
from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional
import math

class LineSpec(BaseModel):
    title: str = "Line Plot"
    x: List[float]
    y: List[float]
    x_label: str = "x"
    y_label: str = "y"
    alt_text: Optional[str] = None
    # optional: if this came from “y = …” text
    function_str: Optional[str] = None

    @model_validator(mode="after")
    def _len_match(self):
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length.")
        if len(self.x) < 2:
            raise ValueError("Need at least 2 points for a line plot.")
        return self

    @field_validator("x", "y")
    @classmethod
    def _finite(cls, v: list[float]) -> list[float]:
        for val in v:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError("All x/y values must be finite numbers.")
        return v

    @model_validator(mode="after")
    def _alt(self):
        if not self.alt_text:
            self.alt_text = f"Line plot of {self.title}."
        return self
