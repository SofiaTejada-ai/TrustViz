from typing import Mapping, Any
from .bar_spec import BarSpec

MAX_CATS = 30
MAX_VAL  = 1e9

def verify_bar(raw: Mapping[str, Any]) -> BarSpec:
    """
    Validate and lightly sanitize a proposed BarSpec dict.
    - ensures matching lengths, sensible sizes
    - clamps numeric weirdness
    """
    spec = BarSpec(**raw)  # schema checks

    # clamp y to sane numeric bounds
    y = []
    for v in spec.y:
        if v != v:          # NaN
            v = 0.0
        if v == float("inf"):
            v = MAX_VAL
        if v == float("-inf"):
            v = -MAX_VAL
        y.append(float(v))
    spec.y = y

    # cap categories (keep first MAX_CATS)
    if len(spec.x) > MAX_CATS:
        spec.x = spec.x[:MAX_CATS]
        spec.y = spec.y[:MAX_CATS]

    return spec
