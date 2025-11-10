# trustviz/rules/core.py
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Step:
    id: str
    event: str                 # e.g., "OutlookPreview"
    where: Dict[str, str]      # simple equality filters
    cite: List[str] = field(default_factory=list)

@dataclass
class SequenceRule:
    name: str
    window_s: int              # e.g., 300
    on: List[str]              # join keys, e.g. ["user","device"]
    steps: List[Step]
    tags: List[str] = field(default_factory=list)

def lint_rule(rule: SequenceRule) -> List[str]:
    errs: List[str] = []
    if not rule.name.strip(): errs.append("name: required")
    if rule.window_s <= 0: errs.append("window_s: must be > 0")
    if not rule.on: errs.append("on: at least one join key")
    if len(rule.steps) < 2: errs.append("steps: need at least 2 in sequence")
    seen = set()
    for i, s in enumerate(rule.steps):
        if not s.id or s.id in seen: errs.append(f"steps[{i}].id: missing/duplicate")
        seen.add(s.id)
        if not s.event: errs.append(f"steps[{i}].event: required")
        if not isinstance(s.where, dict): errs.append(f"steps[{i}].where: object required")
    return errs
