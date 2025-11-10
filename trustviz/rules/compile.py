# trustviz/rules/compile.py
from typing import Dict
from .core import SequenceRule

def to_sigma(rule: SequenceRule) -> Dict:
    detection: Dict[str, Dict] = {}
    sel_names = []
    for i, s in enumerate(rule.steps):
        sel = {"EventID|endswith": s.event}
        for k, v in (s.where or {}).items():
            sel[k] = v
        sel_name = f"sel{i+1}"
        detection[sel_name] = sel
        sel_names.append(sel_name)

    detection["condition"] = "sequence " + " | ".join(sel_names)
    return {
        "title": rule.name,
        "status": "experimental",
        "tags": rule.tags,
        "logsource": {"product": "generic", "service": "custom"},
        "detection": detection,
        "fields": rule.on,
        "falsepositives": ["environmental noise"],
        "level": "high",
        "x_scholarviz_window_s": rule.window_s,
        "x_scholarviz_join": rule.on,
    }

def to_kql(rule: SequenceRule, table: str = "SecurityEvents") -> str:
    lines = [f"let window = {rule.window_s}s;"]
    alias_prev = None
    for i, s in enumerate(rule.steps):
        alias = f"s{i+1}"
        conds = [f"Event endswith '{s.event}'"]
        for k, v in (s.where or {}).items():
            conds.append(f"{k} == '{v}'")
        where_clause = " and ".join(conds) if conds else "true"
        block = [
            f"{alias} =",
            f"  {table}",
            f"  | where {where_clause}",
            f"  | project ts=TimeGenerated, Event, " +
            ", ".join(sorted(set(rule.on + list((s.where or {}).keys())))) + ";",
        ]
        lines += block
        if alias_prev:
            # join on keys and time window
            join_eq = " and ".join([f"$left.{k} == $right.{k}" for k in rule.on])
            lines.append(
                f"{alias}_chain = {alias_prev}"
                f" | join kind=inner ({alias}) on {join_eq}"
                f" | where {alias}.ts between ($left.ts .. $left.ts + window)"
            )
            alias_prev = f"{alias}_chain"
        else:
            alias_prev = alias
    lines.append(f"{alias_prev}")
    return "\n".join(lines)
