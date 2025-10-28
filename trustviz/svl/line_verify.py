# trustviz/svl/line_verify.py
from pydantic import BaseModel, field_validator
from typing import List
import math

class LineSpec(BaseModel):
    kind: str = "line"
    title: str = "Line Plot"
    x: List[float]
    y: List[float]
    x_label: str = "x"
    y_label: str = "y"
    alt_text: str = "Line plot"
    function_str: str | None = None

    @field_validator("x")
    @classmethod
    def _x_ok(cls, v):
        if not isinstance(v, list) or len(v) < 2:
            raise ValueError("x must be a list with length ≥ 2")
        for a in v:
            if not isinstance(a, (int, float)) or not math.isfinite(a):
                raise ValueError("x contains non-finite values")
        return v

    @field_validator("y")
    @classmethod
    def _y_ok(cls, v):
        if not isinstance(v, list) or len(v) < 2:
            raise ValueError("y must be a list with length ≥ 2")
        for a in v:
            if not isinstance(a, (int, float)) or not math.isfinite(a):
                raise ValueError("y contains non-finite values")
        return v

def verify_line(raw: dict) -> LineSpec:
    # Basic length check here so you get a clean error instead of 500
    x, y = raw.get("x"), raw.get("y")
    if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must be lists of the same length (≥ 2)")
    return LineSpec(**raw)
