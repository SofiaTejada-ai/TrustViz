# trustviz/notebook/templates.py
from __future__ import annotations
from typing import Dict, Any, List
import json, re

class ParamError(ValueError): pass

LABS: Dict[str, dict] = {
    "flow_dag_challenge": {
        "title": "Flow DAG challenge",
        "description": "Toggle a back-edge and see how it breaks DAG validity.",
        "params": {
            "add_backedge": {"type": "boolean", "default": False},
        },
        "cells": [
            # Uses a small spec we control; you can pipe the current diagram spec from Studio too.
            """
import json, networkx as nx
ARTIFACTS['checks'] = {}
spec = {
  "nodes": [{"id":"A"},{"id":"B"},{"id":"C"}],
  "edges": [["A","B",{}], ["B","C",{}]],
  "defenses": [{"attachTo":"B","label":"MFA"}]
}
if {add_backedge}:
    spec["edges"].append(["C","A", {}])  # makes a cycle

G = nx.DiGraph()
for n in spec["nodes"]:
    G.add_node(n["id"])
for e in spec["edges"]:
    G.add_edge(e[0], e[1])

checks = ARTIFACTS['checks']
checks['is_dag'] = nx.is_directed_acyclic_graph(G)
checks['edge_count'] = G.number_of_edges()
checks['node_count'] = G.number_of_nodes()
checks['defense_count'] = len(spec.get('defenses', []))

verdicts = []
if not checks['is_dag']: verdicts.append('Flow contains cycles.')
if checks['defense_count'] == 0: verdicts.append('No defensive controls attached.')
ARTIFACTS['verdicts'] = verdicts or ['OK']
"""
        ]
    },

    "roc_builder": {
        "title": "ROC/AUC builder",
        "description": "Paste positives/negatives and compute ROC + AUC.",
        "params": {
            "positives": {"type":"list_float","default":[0.9,0.8,0.7,0.2]},
            "negatives": {"type":"list_float","default":[0.6,0.5,0.4,0.1]},
        },
        "cells": ["""
pos = {positives}
neg = {negatives}
N = len(pos)*len(neg)
if N == 0:
    raise ValueError("Need positives and negatives.")
greater = sum(1 for p in pos for n in neg if p > n)
equal = sum(1 for p in pos for n in neg if p == n)
auc = (greater + 0.5*equal) / N
ARTIFACTS['roc'] = {'pos': len(pos), 'neg': len(neg), 'auc': round(auc, 4)}
ARTIFACTS['verdicts'] = ['OK']
"""]
    },

    "function_sampler": {
        "title": "Function sampler",
        "description": "Sample y=f(x) safely over a range.",
        "params": {
            "expr": {"type":"string","default":"sin(x)"},
            "xmin": {"type":"float","default":0.0, "min":-100, "max":100},
            "xmax": {"type":"float","default":6.28, "min":-100, "max":100},
            "n":    {"type":"int",  "default":200, "min":10, "max":2000},
        },
        "cells": [r"""
import math as _m
expr = "{expr}"
xmin = float({xmin}); xmax = float({xmax}); n = int({n})
if not (-100 <= xmin < xmax <= 100): raise ValueError("Bad range.")
if not (10 <= n <= 2000): raise ValueError("Bad sample size.")
env = {"sin":_m.sin,"cos":_m.cos,"tan":_m.tan,"exp":_m.exp,"log":_m.log,"sqrt":_m.sqrt,"abs":abs,"pi":_m.pi,"e":_m.e}
s = expr.replace("^","**").replace("sin","_m.sin").replace("cos","_m.cos").replace("tan","_m.tan")
xs, ys = [], []
step = (xmax - xmin) / (n-1)
for i in range(n):
    x = xmin + i * step
    try:
        y = eval(s, {"__builtins__":{},"_m":_m,"x":x})
        if isinstance(y,(int,float)) and _m.isfinite(y):
            xs.append(float(x)); ys.append(float(y))
    except Exception: pass
ARTIFACTS['series'] = {"x": xs[:1000], "y": ys[:1000], "expr": expr}
ARTIFACTS['verdicts'] = ['OK'] if len(xs) >= 10 else ['Too few valid samples']
"""]
    }
}

def _coerce(name: str, spec: dict, val: Any):
    t = spec["type"]
    if t == "boolean":
        return bool(val)
    if t == "float":
        v = float(val)
        if "min" in spec and v < spec["min"]: raise ParamError(f"{name} < min")
        if "max" in spec and v > spec["max"]: raise ParamError(f"{name} > max")
        return v
    if t == "int":
        v = int(val)
        if "min" in spec and v < spec["min"]: raise ParamError(f"{name} < min")
        if "max" in spec and v > spec["max"]: raise ParamError(f"{name} > max")
        return v
    if t == "string":
        s = str(val)
        if len(s) > 200: raise ParamError(f"{name} too long")
        return s
    if t == "list_float":
        if isinstance(val, str):
            # split on commas/spaces
            parts = re.split(r"[,\s]+", val.strip()); val = [p for p in parts if p]
        if not isinstance(val, list): raise ParamError(f"{name} must be list")
        out = []
        for x in val:
            out.append(float(x))
        return out
    raise ParamError(f"Unsupported type: {t}")

def render_cells(lab_id: str, params: Dict[str, Any]) -> List[str]:
    lab = LABS.get(lab_id)
    if not lab: raise KeyError("lab not found")
    # coerce/validate params
    validated = {}
    for pname, pspec in lab["params"].items():
        incoming = params.get(pname, pspec.get("default"))
        validated[pname] = _coerce(pname, pspec, incoming)
    # format into cells
    cells = []
    for c in lab["cells"]:
        cells.append(c.format(**validated))
    return cells

def list_labs() -> List[dict]:
    return [
        {"id": k, "title": v["title"], "description": v["description"], "params": v["params"]}
        for k, v in LABS.items()
    ]
