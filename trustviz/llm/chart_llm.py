# trustviz/llm/chart_llm.py
# Same functionality as before, now using Gemini helpers (no OpenAI client).

import json, math, re
from typing import Literal, Optional
from trustviz.llm.gemini_client import gen_text, gen_json

ChartKind = Literal["bar", "pie", "line"]

# --- Narrative + viz segments guidance (matching your UI) ---
SYSTEM_SEGMENTS = (
  "You are producing a SHORT step-by-step walkthrough for a chart request. "
  "Respond with ONLY JSON using keys: title, labels, segments. "
  "labels: array of strings that appear in the chart (e.g., series names, categories). "
  "segments: ordered array mixing text and viz objects. "
  "For text: {\"type\":\"text\",\"content\":\"1–3 sentences, educational, defense-only\"}. "
  "For viz: {\"type\":\"viz\",\"mode\":\"chart\",\"chart\":{...}} where chart matches one of: "
  "{kind:'bar', title, x:[str], y:[number], x_label, y_label, alt_text} OR "
  "{kind:'pie', title, labels:[str], values:[number], alt_text} OR "
  "{kind:'line', title, x:[number], y:[number], x_label, y_label, alt_text, function_str?}. "
  "Numbers must be finite; arrays length 2..50. No code fences."
)

# Exact rules we send to Gemini for a single chart JSON spec.
SPEC_SYSTEM = (
  "Return ONLY JSON for a single chart. "
  "Allowed schema keys: kind('bar'|'pie'|'line'), title, x?, y?, labels?, values?, "
  "x_label?, y_label?, alt_text, function_str?. "
  "Numbers must be finite; arrays length 2..50. No prose, no code fences."
)

# Rules for generating the two short paragraphs (intro/outro).
EXPLAIN_SYSTEM = (
  "You write concise, student-friendly explanations for charts. "
  "Return ONLY JSON with keys: intro, outro. "
  "Each should be 1–3 sentences. Educational and defense-only tone. "
  "No prose outside JSON. No code fences."
)


# ----------------------------- Public API -----------------------------

def get_chart_walkthrough(q: str, model: Optional[str]) -> dict:
    """
    Returns:
      {
        "segments": [
          {"type":"text","content": "..."},
          {"type":"viz","mode":"chart","chart": {...}},
          {"type":"text","content": "..."}
        ],
        "labels": [...]
      }
    """
    # 1) Get a chart spec (LLM or deterministic fallback)
    try:
        spec = _try_llm_chart_json(q, model)
    except Exception:
        spec = _fallback_chart(q)

    # 2) Collect labels for bolding
    labels: list[str] = []
    k = (spec or {}).get("kind")
    if k == "bar":
        labels = list(spec.get("x", []) or [])
    elif k == "pie":
        labels = list(spec.get("labels", []) or [])
    elif k == "line":
        for key in ("title", "x_label", "y_label", "function_str"):
            v = spec.get(key)
            if isinstance(v, str) and v.strip():
                labels.append(v.strip())

    # 3) Two short paragraphs (defense-only tone)
    intro = "This chart summarizes the requested data at a glance."
    outro = "Use this view for discussion and decision-making; no sensitive procedures are exposed."
    if model:
        try:
            user = (
                f"User prompt: {q}\n"
                f"Chart kind: {k}\n"
                f"Terms to weave in if natural: {', '.join(labels) if labels else '(none)'}"
            )
            expl = gen_json(EXPLAIN_SYSTEM, user, model=model) or {}
            if isinstance(expl, dict):
                intro = (expl.get("intro") or intro).strip()
                outro = (expl.get("outro") or outro).strip()
        except Exception:
            pass  # keep defaults

    return {
        "segments": [
            {"type": "text", "content": intro},
            {"type": "viz",  "mode": "chart", "chart": spec},
            {"type": "text", "content": outro},
        ],
        "labels": labels,
    }


def get_chart_spec_from_text(q: str, model: Optional[str]) -> dict:
    """Single-chart spec (tries LLM, falls back deterministically)."""
    try:
        return _try_llm_chart_json(q, model)
    except Exception:
        return _fallback_chart(q)


# ----------------------------- Internals ------------------------------

def _try_llm_chart_json(q: str, model: Optional[str]) -> dict:
    """
    Gemini-backed JSON chart spec.
    Kept behavior: if LLM fails or returns non-JSON, caller falls back.
    """
    if not model:
        raise RuntimeError("No model defined")
    user = f"User query: {q}"
    data = gen_json(SPEC_SYSTEM, user, model=model) or {}
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object")
    return data


# Heuristic, safe fallback (works w/o LLM) — unchanged behavior
def _fallback_chart(q: str) -> dict:
    s = (q or "").lower()
    if "pie" in s:
        return {
            "kind":"pie",
            "title":"Protocol distribution",
            "labels":["HTTP","HTTPS","SSH","Other"],
            "values":[40, 35, 15, 10],
            "alt_text":"Pie showing distribution of protocols.",
        }
    if "bar" in s:
        return {
            "kind":"bar",
            "title":"Incidents by severity",
            "x":["Low","Medium","High","Critical"],
            "y":[15, 9, 4, 2],
            "x_label":"Severity",
            "y_label":"Count",
            "alt_text":"Bar chart of incidents by severity.",
        }
    # function plot: “plot y = sin(x) from 0 to 6.28”
    m = re.search(r"y\s*=\s*([a-z0-9\^\*\+\-\/\(\)\.\s]+).*?(\d+(\.\d+)?)\s*to\s*(\d+(\.\d+)?)", s)
    if ("line" in s or "plot" in s or "function" in s or m):
        x0, x1 = 0.0, 10.0
        func = "x"
        if m:
            func = m.group(1)
            x0 = float(m.group(2))
            x1 = float(m.group(4))
        xs, ys = [], []
        n = 50
        for i in range(n):
            x = x0 + (x1 - x0)*i/(n-1)
            expr = func.replace("^","**")
            expr = expr.replace("sin","math.sin").replace("cos","math.cos").replace("tan","math.tan")
            y = eval(expr, {"__builtins__":{},"math":math,"x":x})
            xs.append(float(x)); ys.append(float(y))
        return {
            "kind":"line",
            "title":f"y = {func}",
            "x": xs, "y": ys,
            "x_label":"x","y_label":"y",
            "alt_text":"Line plot of the function over the requested interval.",
            "function_str": f"y={func}",
        }
    # default bar
    return {
        "kind":"bar",
        "title":"Counts by category",
        "x":["A","B","C","D"],
        "y":[3,6,2,5],
        "x_label":"Category","y_label":"Count",
        "alt_text":"Bar chart of counts by category.",
    }
