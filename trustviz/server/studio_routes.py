# trustviz/server/studio_routes.py — Studio routes + UI (diagram-first with fallbacks)

import os, re, json, html, traceback, hashlib, math
from typing import Tuple, List
from fastapi import APIRouter, Query, Request, Body
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

# Verifiers/adapters (deterministic, fast)
from trustviz.svl.bar_verify import verify_bar
from trustviz.svl.pie_verify import verify_pie
from trustviz.svl.line_verify import verify_line
from trustviz.svl.graph_verify import KnowledgeGraphSpec, verify_kg
from trustviz.ve.graph_adapter import kg_cyto_payload

# LLM adapters (Gemini-backed)
from trustviz.llm.chart_llm import get_chart_spec_from_text, get_chart_walkthrough
from trustviz.llm.kg_llm import get_llm_kg_walkthrough
from trustviz.llm.diagram_llm import get_diagram_walkthrough

# Optional notebook helper (for server-side validation run)
from trustviz.notebook.runner import NotebookRunner
from trustviz.llm.gemini_client import gen_json  # used for captions and Lab explanations
from fastapi.responses import Response
from trustviz.ai.edu_charts import render_chart_png
router = APIRouter()

# ------------------------- Globals / Config -------------------------
MODEL = os.environ.get("TRUSTVIZ_LLM_MODEL", "gemini-2.5-pro")
_EXPLAIN_RULES = (
  "You must return a single JSON object with EXACTLY these keys:\n"
  "intro, what_you_see, how_it_works, why_it_matters, callouts, glossary\n"
  "- intro: 2–3 sentences beginner-friendly (min 50 words).\n"
  "- what_you_see: 1 paragraph naming 3–5 boxes/arrows by label (min 70 words).\n"
  "- how_it_works: 1 paragraph mapping controls to stages (min 90 words).\n"
  "- why_it_matters: 1 paragraph tied to SOC/audits/outcomes (min 70 words).\n"
  "- callouts: ordered list of 3–6 bullets, each '<Label>: takeaway'.\n"
  "- glossary: 3–6 objects {'term': .., 'def': ..} (11–25 words each).\n"
  "Rules: defense-only; no offensive procedures. Prefer exact diagram labels. No filler."
)

# LLM rule for Lab explanations
_EXPLAIN_ARTIFACTS_RULES = (
  "Write one short, student-friendly explanation (plain text, no JSON).\n"
  "Start with a one-sentence summary of the flow.\n"
  "Then explain each verdict plainly and suggest one concrete fix per issue.\n"
  "If verdicts == ['OK'], give one tip to strengthen the model anyway.\n"
  "Keep it defense-only; do not include any offensive/operational steps."
)

# LLM rule for the optional “Ask about this diagram”
_ASK_RULES = (
  "Defense-only tutoring. Use exact diagram labels when helpful.\n"
  "Answer concisely (<=170 words). If the question is vague, briefly say what the diagram shows first,\n"
  "then give 2–4 concrete insights or next steps a defender should consider.\n"
  "Never provide offensive instructions."
)

_FUNC = r"(sin|cos|tan|exp|log|sqrt)"
PI_ALIASES = {"π": math.pi, "pi": math.pi}

# Show/hide the Notebook “Learning Lab” box in the UI
SHOW_LAB = True

# Course-topic presets (Google Cybersecurity cert-aligned prompts)
TOPIC_PRESETS = {
  "Foundations": "Explain the CIA triad and give a high-level security lifecycle diagram.",
  "Linux & Sysadmin": "Show the Linux auth flow and where logs land; add controls.",
  "Networking": "Layered model (L2–L7) and common controls at each layer.",
  "Security Tools & SIEM": "End-to-end log pipeline to SIEM with detections and triage steps.",
  "Threat Modeling (ATVR)": "Identify assets, threats, vulns, and controls; show a risk path.",
  "Incident Response": "NIST IR lifecycle with defender actions and artifacts at each phase.",
  "Access Management": "MFA + SSO + least privilege + token risks and mitigations.",
  "Cloud Security": "Public bucket misconfig to exfil path with cloud-native controls.",
}

# --------------------------- Safety & Reframe ---------------------------
_DENYLIST = [
    r"\b(ddos|dos)\b", r"\b(botnet|c2|command\s*and\s*control)\b",
    r"\b(keylogger|rootkit|rat)\b", r"\bzero[-\s]?day\b",
    r"\bexploit\s+code\b", r"\bcredential\s+stuffing\b",
    r"\bphishing\s+kit\b", r"\bpassword\s*cracker\b",
    r"\bport\s*scan\s*(script|code)\b", r"\bsql\s*injection\s*(payload|exploit)\b",
    r"\bbypass\s+mfa\b", r"\breal\s+bank\s+site\b",
]
def _is_risky(text: str) -> bool:
    s = (text or "").lower()
    return any(re.search(p, s) for p in _DENYLIST)

def _defensive_reframe(text: str):
    return (
        "This view is reframed for safety and education only. It shows common attack paths and where defenders can stop them—no procedural details.",
        {"reframed": True, "reason": "operational/sensitive intent"}
    )

# --------------------------- Intent routing ---------------------------
_MATH_PAT = re.compile(r"(?:^|\s)(draw|plot|visualize)\s+|^y\s*=|^f\s*\(\s*x\s*\)\s*=", re.I)
def _looks_like_function(text: str) -> bool:
    s = text or ""
    return bool(_MATH_PAT.search(s)) and ("x" in s or "y" in s or "f(" in s)

def _looks_like_roc(text: str) -> bool:
    s = (text or "").lower()
    return (" roc" in s) or ("auc" in s) or ("positives" in s and "negatives" in s)

def _route_intent(q: str) -> str:
    s = (q or "").lower()
    if any(k in s for k in ["knowledge graph", "attack graph", "attack-graph", "graph of", "graph:"]):
        return "graph"
    if _looks_like_roc(s): return "roc"
    if any(t in s for t in ["how ", "why ", "happen", "steps", "process", "flow", "lifecycle", "kill chain"]):
        return "diagram"
    if _looks_like_function(q) or any(k in s for k in ["bar", "pie", "line", "plot", "chart", "histogram", "draw"]):
        return "chart"
    return "diagram"

# ------------------------- Topic mapping (extend over time) -------------------------
def _topic_for(q: str) -> str:
    s = (q or "").lower()
    if "phish" in s: return "phishing"
    if "cia triad" in s or ("confidentiality" in s and "integrity" in s and "availability" in s): return "cia"
    if "risk" in s and "triage" in s: return "risk"
    if "zero trust" in s or "zero-trust" in s: return "zerotrust"
    if any(k in s for k in ["oauth", "token"]): return "oauth"
    if any(k in s for k in ["cloud", "s3", "bucket", "misconfig"]): return "cloud"
    if any(k in s for k in ["supply", "sbom", "software supply"]): return "supply"
    return "breach"

# ------------------------- Mermaid safety -------------------------
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]()_:.-> \n|/%'\",+&?-")
def _sanitize_mermaid(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch in _ALLOWED)

def _default_mermaid(title: str = "Security Lifecycle") -> str:
    return (
      "flowchart LR\n"
      "  A[Ask a security question] --> B[Conceptual diagram]\n"
      "  B --> C[Key defensive controls]\n"
      "  C --> D[Safer operations]\n"
    )

# ------------------------- Diagram helpers -------------------------
def _bold_terms(text: str, labels: list[str]) -> str:
    s = html.escape(text or "")
    phrases = sorted({l.strip() for l in (labels or []) if l}, key=len, reverse=True)
    for p in phrases:
        pat = re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE)
        s = pat.sub(lambda m: f"<strong>{html.escape(m.group(0))}</strong>", s)
    return s

def _chart_spec_to_plotly(chart: dict) -> dict:
    kind = (chart or {}).get("kind")
    title = chart.get("title") or ""
    if kind == "pie":
        return {"data":[{"type":"pie","labels":chart.get("labels",[]),"values":chart.get("values",[])}],
                "layout":{"title":title}}
    if kind == "bar":
        return {"data":[{"type":"bar","x":chart.get("x",[]),"y":chart.get("y",[])}],
                "layout":{"title":title,"yaxis":{"title":chart.get("y_label") or "Value"}}}
    if kind == "line":
        return {"data":[{"type":"scatter","mode":"lines","x":chart.get("x",[]),"y":chart.get("y",[])}],
                "layout":{"title":title,"xaxis":{"title":chart.get("x_label") or "x"},
                                      "yaxis":{"title":chart.get("y_label") or "y"}}}
    return {"data":[], "layout":{"title": title or "Chart"}}

def _mermaid_template(kind: str) -> str:
    if kind == "breach":
        return """
flowchart LR
  A[Recon] --> B[Initial Access]
  B --> C[Privilege Escalation & Lateral Movement]
  C --> D[Data Discovery & Collection]
  D --> E[Exfiltration]
  E --> F[Monetization / Impact]
  D1[[MFA]] -.-> B
  D2[[EDR/AV]] -.-> C
  D3[[DLP]] -.-> D
  D4[[Egress Monitor]] -.-> E
""".strip()
    if kind == "phishing":
        return """
flowchart LR
  P1[Recon/Targeting] --> P2[Phish Email/SMS]
  P2 --> P3[User Interaction]
  P3 --> P4[Token/Cred Steal]
  P4 --> P5[Initial Access]
  P5 --> P6[Lateral Movement]
  P6 --> P7[Data Access/Exfil]
  D1[[Email Filter + DMARC]] -.-> P2
  D2[[MFA/Phishing-Resistant]] -.-> P3
  D3[[EDR/Least Privilege]] -.-> P5
  D4[[DLP/Egress Monitor]] -.-> P7
""".strip()
    if kind == "supply":
        return """
flowchart LR
  S1[Compromise Vendor/Build] --> S2[Malicious Update]
  S2 --> S3[Customer Deployment]
  S3 --> S4[Privilege Escalation]
  S4 --> S5[Lateral Movement/Data]
  D1[[Code Signing/Verify]] -.-> S2
  D2[[Allow-List/Runtime Monitor]] -.-> S3
  D3[[Segmentation/EDR]] -.-> S4
""".strip()
    if kind == "cloud":
        return """
flowchart LR
  C1[Public Bucket/Misconfig] --> C2[Unauth Access]
  C2 --> C3[Data Enumeration]
  C3 --> C4[Bulk Download]
  C4 --> C5[Monetization/Leak]
  D1[[IAM Least Privilege]] -.-> C2
  D2[[Block Public Access]] -.-> C1
  D3[[DLP/Object Lock]] -.-> C4
""".strip()
    if kind == "oauth":
        return """
flowchart LR
  O1[Phish/OAuth Consent] --> O2[Token Issued]
  O2 --> O3[Token Theft/Replay]
  O3 --> O4[API/Data Access]
  D1[[PKCE/Bound Tokens]] -.-> O2
  D2[[Short TTL/Rotation]] -.-> O3
  D3[[Anomaly Detection]] -.-> O4
""".strip()
    if kind == "zerotrust":
        return """
flowchart LR
  Z1[Request] --> Z2[Policy: Identity+Device+Context]
  Z2 --> Z3[Grant with Least Priv]
  Z3 --> Z4[Continuous Verify]
  Z4 --> Z5[Re-evaluate/Block]
  D1[[MFA/Device Posture/Network Microseg]] -.-> Z2
""".strip()
    if kind == "auth_mfa":
        return """
flowchart LR
  A[User Sign-in] --> B[MFA Challenge]
  B -->|consent fatigue/phish| C[Token Theft/Replay]
  B -->|SIM swap/OTP intercept| D[OTP Compromise]
  C --> E[Session Access]
  D --> E
  E --> F[Lateral Movement/Data]
  D1[[FIDO2/WebAuthn]] -.-> B
  D2[[Number matching / rate limits]] -.-> B
  D3[[Token binding / short TTL / rotation]] -.-> C
  D4[[Device posture + EDR]] -.-> E
  D5[[Anomaly detection / CA policies]] -.-> E
""".strip()
    return _default_mermaid()

_KNOWN_KINDS = {"breach","phishing","supply","cloud","oauth","zerotrust","auth_mfa"}

def _diagram_for(q: str) -> dict:
    kind = _topic_for(q)
    base = _mermaid_template(kind) if kind in _KNOWN_KINDS else _default_mermaid()
    mermaid = _sanitize_mermaid(base)
    spec = {
        "nodes": [{"id":"A","label":"Recon"},{"id":"B","label":"Initial Access"}],
        "edges": [["A","B", {"label":"progression"}]],
        "defenses": [{"attachTo":"B","label":"MFA"}],
    }
    return {"mode": "diagram", "mermaid": mermaid, "bullets": [kind], "spec": spec}

# ------------------------- Small parsers -------------------------
_PAIR_SPLIT = re.compile(r"[;,]\s*")
def _parse_pairs(text: str):
    labels, vals = [], []
    for tok in _PAIR_SPLIT.split(text.strip()):
        if not tok: continue
        piece = tok.rsplit(":", 1)[-1].strip()
        m = re.match(r"(.*?\S)\s+(-?\d+(?:\.\d+)?)\s*$", piece)
        if m:
            labels.append(m.group(1).strip()); vals.append(float(m.group(2)))
    return labels, vals

_RANGE_RE = re.compile(r"\[\s*([^\],]+)\s*,\s*([^\],]+)\s*\]")
def _parse_range(text: str) -> Tuple[float, float] | None:
    m = _RANGE_RE.search(text)
    if not m: return None
    def num(s: str) -> float:
        s = s.strip().lower().replace("−", "-")
        for k, v in PI_ALIASES.items(): s = s.replace(k, str(v))
        return float(eval(s, {"__builtins__": {}}, {"pi": math.pi}))
    return num(m.group(1)), num(m.group(2))

_PARAM_RE = re.compile(r"([A-Za-z])\s*=\s*([-+]?\d+(?:\.\d+)?)")
def _apply_params(expr: str, text: str) -> str:
    params = {m.group(1): m.group(2) for m in _PARAM_RE.finditer(text)}
    for k, v in params.items(): expr = re.sub(rf"\b{k}\b", v, expr)
    return expr

def _normalize_expr(expr: str) -> str:
    s = (expr or "").strip()
    s = re.sub(r'^\s*(draw|plot|graph|visualize|compute)\s+', '', s, flags=re.I)
    s = re.sub(r'^\s*y\s*=\s*', '', s, flags=re.I)
    s = re.sub(r'^\s*f\s*\(\s*x\s*\)\s*=\s*', '', s, flags=re.I)
    s = _apply_params(s, expr)
    s = s.replace("^", "**")
    s = re.sub(rf"\b{_FUNC}\s*x\b", r"\1(x)", s, flags=re.I)
    s = re.sub(rf"(\d)\s*({_FUNC})\b", r"\1*\2", s, flags=re.I)
    s = re.sub(r"(\d)\s*x\b", r"\1*x", s)
    s = re.sub(r"\bx\s*(\d)", r"x*\1", s)
    s = re.sub(r"\bx\s*\(", "x*(", s)
    s = re.sub(r"\)\s*x\b", ")*x", s)
    s = re.sub(r"\s+", "", s)
    return s

def _sample_function(expr: str, xmin=-5.0, xmax=5.0, n=600):
    import math as _m
    env = {"sin": _m.sin, "cos": _m.cos, "tan": _m.tan, "exp": _m.exp, "log": _m.log,
           "sqrt": _m.sqrt, "abs": abs, "pi": _m.pi, "e": _m.e}
    s = _normalize_expr(expr)
    xs, ys = [], []
    step = (xmax - xmin) / max(n - 1, 1)
    for i in range(n):
        x = xmin + i * step
        env["x"] = x
        try:
            y = eval(s, {"__builtins__": {}}, env)
            if isinstance(y, (int, float)) and _m.isfinite(y):
                xs.append(float(x)); ys.append(float(y))
        except Exception:
            continue
    if len(xs) < 10:
        raise ValueError(f"Not enough valid samples to plot “{expr}”. Try a different range.")
    return {
        "data": [{"type": "scatter", "mode": "lines", "x": xs, "y": ys, "name": f"y = {s}"}],
        "layout": {"title": f"y = {s}", "xaxis": {"title": "x"}, "yaxis": {"title": "y"}}
    }

# ------------------------- ROC helper -------------------------
def _roc_from_lists(pos: List[float], neg: List[float]):
    try:
        from sklearn.metrics import roc_curve, auc
        import numpy as np
        y_true = np.array([1]*len(pos) + [0]*len(neg))
        y_score = np.array(pos + neg)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        A = float(auc(fpr, tpr))
        return {"data": [{"type":"scatter","mode":"lines","x":fpr.tolist(),"y":tpr.tolist(),"name":"ROC"}],
                "layout":{"title":f"ROC (AUC={A:.3f})","xaxis":{"title":"FPR"},"yaxis":{"title":"TPR"}}}, A
    except Exception:
        N = len(pos)*len(neg)
        if N == 0: raise ValueError("Need positives and negatives.")
        greater = sum(1 for p in pos for n in neg if p > n)
        equal = sum(1 for p in pos for n in neg if p == n)
        auc_val = (greater + 0.5*equal) / N
        pts = sorted([(p,1) for p in pos] + [(n,0) for n in neg], key=lambda x: -x[0])
        tp = fp = 0; P=len(pos); Nn=len(neg)
        xs=[0.0]; ys=[0.0]
        for _,y in pts:
            tp += (y==1); fp += (y==0)
            xs.append(fp/max(Nn,1)); ys.append(tp/max(P,1))
        return {"data":[{"type":"scatter","mode":"lines","x":xs,"y":ys,"name":"ROC"}],
                "layout":{"title":f"ROC (AUC≈{auc_val:.3f})","xaxis":{"title":"FPR"},"yaxis":{"title":"TPR"}}}, auc_val

def _parse_roc(text: str) -> Tuple[List[float], List[float]]:
    ps = re.search(r"positives?\s*([0-9\.,\s-]+)", text, re.I)
    ns = re.search(r"negatives?\s*([0-9\.,\s-]+)", text, re.I)
    if not (ps and ns): raise ValueError("Provide 'positives ...; negatives ...' lists.")
    P = [float(x) for x in re.split(r"[,\s]+", ps.group(1).strip()) if x]
    N = [float(x) for x in re.split(r"[,\s]+", ns.group(1).strip()) if x]
    return P, N

# ------------------------- Notebook validation helper -------------------------
import json as _json_mod
def _validate_flow_spec(spec: dict) -> dict:
    """Returns artifacts like {'checks': {...}, 'verdicts': [...]} via Notebook runner."""
    cells = [f"""
import json, networkx as nx
ARTIFACTS['checks'] = {{}}
spec = json.loads('''{_json_mod.dumps(spec)}''')
G = nx.DiGraph()
for n in spec.get('nodes', []): G.add_node(n.get('id'))
for e in spec.get('edges', []):
    if isinstance(e, (list,tuple)) and len(e)>=2: G.add_edge(e[0], e[1])
checks = ARTIFACTS['checks']
checks['node_count'] = G.number_of_nodes()
checks['edge_count'] = G.number_of_edges()
checks['is_dag'] = nx.is_directed_acyclic_graph(G)
checks['isolated_nodes'] = [n for n in G.nodes() if G.degree(n)==0]
checks['defense_count'] = len(spec.get('defenses', []) or [])
verdicts = []
if not checks['is_dag']: verdicts.append('Flow contains cycles; remove back-edges.')
if checks['defense_count'] == 0: verdicts.append('No defensive controls attached.')
if checks['isolated_nodes']: verdicts.append('Isolated nodes: ' + json.dumps(checks["isolated_nodes"][:5]))
ARTIFACTS['verdicts'] = verdicts or ['OK']
"""]
    runner = NotebookRunner(timeout_s=6)
    try:
        res = runner.run(cells)
        return res.artifacts or {}
    finally:
        runner.stop()

# ------------------------- Provenance & helpers -------------------------
def _prov(q: str, svl: str, extra: dict | None = None) -> dict:
    h = hashlib.sha256((q or "").encode("utf-8")).hexdigest()[:16]
    out = {"inputs_hash": h, "svl_decision": svl}
    if extra: out.update(extra)
    return out

def _labels_from_mermaid(src: str) -> list[str]:
    labs = []
    for m in re.finditer(r"\b[A-Za-z0-9_]+\s*[\[\(]([^\]\)]+)[\]\)]", src or ""):
        t = m.group(1).strip()
        if t and t.lower() not in {"defensive_controls"}:
            labs.append(t)
    out, seen = [], set()
    for t in labs:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _auto_explain(kind: str, labels: list[str]) -> dict:
    defaults = {
        "breach": ["Recon","Initial Access","Privilege Escalation & Lateral Movement",
                   "Data Discovery & Collection","Exfiltration","Monetization / Impact"],
        "phishing": ["Phish Email/SMS","User Interaction","Token/Cred Steal",
                     "Initial Access","Lateral Movement","Data Access/Exfil"],
        "cloud": ["Public Bucket/Misconfig","Unauth Access","Data Enumeration","Bulk Download"],
        "oauth": ["OAuth Consent","Token Issued","Token Theft/Replay","API/Data Access"],
        "supply": ["Compromise Vendor/Build","Malicious Update","Customer Deployment",
                   "Privilege Escalation","Lateral Movement/Data"],
        "zerotrust": ["Request","Policy: Identity+Device+Context","Grant with Least Priv",
                      "Continuous Verify","Re-evaluate/Block"],
    }.get(kind, [])
    names = [n for n in defaults if n in labels] or (labels[:6] or ["Stage A","Stage B","Stage C"])
    intro = ("This diagram summarizes a common security scenario and where layered controls interrupt risk. "
             "It helps learners map phases to the exact points where detections and policies have leverage.")
    what = (f"You are seeing a left-to-right flow. Typical stages include {', '.join(names[:3])}"
            + (", and " + names[3] if len(names) > 3 else "")
            + ". Dashed connectors indicate where controls attach to stop progression.")
    how = (f"The flow begins at {names[0]} and proceeds through "
           f"{', '.join(names[1:max(2,len(names)//2)])}. Controls reduce either the chance of the next step or "
           "the impact if it occurs. MFA hardens Initial Access; EDR/AV constrains escalation; DLP limits sensitive "
           "collection from turning into Exfiltration; egress monitoring spots late-stage anomalies.")
    why = ("This matters because incidents unfold along these stages in tickets and logs. Aligning detections and "
           "playbooks to each stage shortens time-to-detect and time-to-contain—key audit and executive metrics.")
    callouts = [
        f"{names[0]}: define early telemetry to catch targeting or scanning.",
        f"{names[1] if len(names)>1 else 'Initial Access'}: enforce phishing-resistant MFA.",
        f"{names[2] if len(names)>2 else 'Privilege Escalation'}: apply least privilege and monitor elevation.",
        f"{names[3] if len(names)>3 else 'Data Discovery & Collection'}: tag sensitive data and enable DLP.",
    ]
    glossary = [
        {"term":"MFA","def":"A second factor that verifies identity to prevent easy account takeover."},
        {"term":"EDR","def":"Host-level detection/response to spot and contain malicious activity."},
        {"term":"DLP","def":"Policies that monitor and block sensitive data leaving the environment."},
    ]
    return {"intro": intro, "what_you_see": what, "how_it_works": how,
            "why_it_matters": why, "callouts": callouts, "glossary": glossary}

def _to_text(x):
    """Coerce any value to a short, safe text for html.escape."""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

# ------------------------------ API: Learning Lab ------------------------------
@router.post("/lab/validate")
def lab_validate(spec: dict = Body(...)):
    """Run the server-side notebook to validate a diagram spec and return artifacts."""
    try:
        artifacts = _validate_flow_spec(spec or {})
        return JSONResponse({"ok": True, "artifacts": artifacts})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@router.post("/lab/explain")
def lab_explain(payload: dict = Body(...)):
    """Use the LLM to explain notebook validation artifacts in plain language."""
    spec = (payload or {}).get("spec") or {}
    user_q = (payload or {}).get("q") or ""
    try:
        artifacts = _validate_flow_spec(spec)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"validate: {e}"}, status_code=400)

    labels = [n.get("label","") for n in (spec.get("nodes") or []) if n.get("label")]
    try:
        user = (
            "Diagram labels you may reference: " + ", ".join(labels[:12]) + "\n"
            "Validation artifacts JSON:\n" + json.dumps(artifacts) + "\n"
            f"Student question/focus (optional): {user_q}\n"
        )
        text = gen_json(_EXPLAIN_ARTIFACTS_RULES, user, model=MODEL)
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)
        return JSONResponse({"ok": True, "artifacts": artifacts, "explanation": text})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"llm: {e}"}, status_code=500)

# ------------------------------ API: Optional “Ask about this diagram” --------
@router.post("/studio/ask")
def studio_ask(payload: dict = Body(...)):
    """
    Accepts { "mermaid": str, "q": str } and returns a defense-only LLM answer.
    If labels aren't provided, they are auto-extracted from Mermaid.
    """
    mer = (payload or {}).get("mermaid") or _default_mermaid()
    q = (payload or {}).get("q") or ""
    labels = _labels_from_mermaid(mer)
    try:
        user = (
            "Mermaid diagram source:\n" + mer + "\n\n" +
            "Diagram labels you may reference verbatim: " + ", ".join(labels[:20]) + "\n" +
            "Student question:\n" + q
        )
        text = gen_json(_ASK_RULES, user, model=MODEL)
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)
        return JSONResponse({"ok": True, "answer": text})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ------------------------------ /studio ---------------------------------------
@router.get("/edu/chart")
def edu_chart(kind: str = "activations"):
    try:
        png = render_chart_png(kind)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        return Response(content=f"error: {e}", media_type="text/plain", status_code=500)
@router.get("/studio", response_class=HTMLResponse)
def studio(
    request: Request,
    q: str = Query(
        "",
        description="Ask a security 'how/why', a knowledge graph, or a chart/math/ROC request",
    ),
):
    explanation = ""
    error = "-"
    fig = None
    cyto = None

    # Landing
    if not (q or "").strip():
        explanation = ("Pick a topic preset or ask a process question for a diagram; "
                       "type “draw x^2” for a function; or provide positives/negatives for ROC.")
        return HTMLResponse(_studio_shell(q, explanation, error, fig, cyto, mermaid=None, segments_obj=None))

    # Safety: defense-only fallback
    if getattr(request.state, "risky", False) or _is_risky(q):
        intro_text, _provx = _defensive_reframe(q)
        payload = _diagram_for(q)
        mer_src = payload.get("mermaid") or _default_mermaid()
        outro_text = "Notice how controls like MFA, EDR/AV, DLP and Egress monitoring attach to key steps. Follow the arrows to see where defenders can interrupt the path."
        walk = {
            "segments": [
                {"type": "text", "content": intro_text},
                {"type": "viz", "mode": "diagram", "mermaid": mer_src},
                {"type": "text", "content": outro_text},
            ],
            "labels": []
        }
        return HTMLResponse(_studio_shell(q, "Walkthrough with a conceptual diagram.", "-", None, None, mermaid=None, segments_obj=walk))

    # Route intent
    intent = _route_intent(q)

    try:
        # -------- Diagram --------
        if intent == "diagram":
            payload = _diagram_for(q)
            mer = payload.get("mermaid") or _default_mermaid()
            kind = (_topic_for(q) or "breach")
            labels = _labels_from_mermaid(mer)

            # fallback explanation
            expl = _auto_explain(kind, labels)

            # try LLM JSON and merge if strong enough
            try:
                user = (
                    f"User prompt: {q}\n"
                    f"Mermaid diagram source:\n{mer}\n"
                    f"Diagram labels you MAY reference verbatim: {', '.join(labels) if labels else '(none)'}"
                )
                data = gen_json(_EXPLAIN_RULES, user, model=MODEL) or {}
                def ok_text(s, n=120): return isinstance(s, str) and len(s.strip()) >= n
                def ok_list(xs, n=3): return isinstance(xs, list) and len(xs) >= n
                if isinstance(data, dict):
                    if ok_text(data.get("intro"), 80):          expl["intro"] = data["intro"].strip()
                    if ok_text(data.get("what_you_see"), 80):   expl["what_you_see"] = data["what_you_see"].strip()
                    if ok_text(data.get("how_it_works"), 120):  expl["how_it_works"] = data["how_it_works"].strip()
                    if ok_text(data.get("why_it_matters"), 80): expl["why_it_matters"] = data["why_it_matters"].strip()
                    if ok_list(data.get("callouts"), 3):        expl["callouts"] = data["callouts"]
                    if ok_list(data.get("glossary"), 3):        expl["glossary"] = data["glossary"]
            except Exception:
                pass

            # render sections, robust to odd shapes
            def _coerce_callout(x):
                if isinstance(x, str): return x
                if isinstance(x, dict):
                    return x.get("text") or x.get("note") or x.get("label") or x.get("title") \
                           or (f"{x.get('term','')}: {x.get('def','')}".strip(": ").strip()) \
                           or json.dumps(x, ensure_ascii=False)
                return str(x)

            def _glossary_li(g):
                if isinstance(g, dict):
                    term = g.get("term") or g.get("label") or "Term"
                    definition = g.get("def") or g.get("definition") or g.get("text") or ""
                else:
                    term, definition = "Term", str(g)
                return f"<li><strong>{html.escape(term)}:</strong> {html.escape(definition)}</li>"

            callouts_html = "".join(f"<li>{html.escape(_coerce_callout(c))}</li>" for c in (expl.get("callouts") or []))
            glossary_html = "".join(_glossary_li(g) for g in (expl.get("glossary") or []))

            sections_below = [
                {"type": "text", "content": f"<h3>What you’re seeing</h3><p>{html.escape(expl['what_you_see'])}</p>"},
                {"type": "text", "content": f"<h3>How it works</h3><p>{html.escape(expl['how_it_works'])}</p>"},
                {"type": "text", "content": f"<h3>Why it matters</h3><p>{html.escape(expl['why_it_matters'])}</p>"},
                {"type": "text", "content": f"<h3>Key callouts</h3><ol>{callouts_html}</ol>"},
                {"type": "text", "content": f"<h3>Glossary</h3><ul>{glossary_html}</ul>"},
            ]

            segments = [
                {"type": "text", "content": html.escape(expl["intro"])},
                {"type": "viz", "mode": "diagram", "mermaid": mer},
                *sections_below,
            ]

            # if Lab on, append server validation verdicts (quick)
            try:
                if SHOW_LAB and isinstance(payload.get("spec"), dict):
                    artifacts = _validate_flow_spec(payload["spec"])
                    verdicts = artifacts.get("verdicts") or []
                    if verdicts:
                        segments.append({"type": "text", "content": "<p><em>Validation:</em> " + html.escape("; ".join(verdicts)) + "</p>"})
            except Exception:
                pass

            return HTMLResponse(_studio_shell(
                q=q,
                explanation="Walkthrough: rich intro above, detailed caption below the diagram.",
                error="-", fig_json=None, cyto=None, mermaid=None, segments_obj={"segments": segments}
            ))

        # -------- Knowledge Graph --------
        if intent == "graph":
            walk = get_llm_kg_walkthrough(q, MODEL)
            return HTMLResponse(_studio_shell(
                q=q, explanation="Walkthrough with a knowledge graph and short explanations.",
                error="-", fig_json=None, cyto=None, mermaid=None, segments_obj=walk
            ))

        # -------- ROC --------
        if intent == "roc":
            P, N = _parse_roc(q)
            fig, auc_val = _roc_from_lists(P, N)
            explanation = f"ROC computed from your lists. AUC = {auc_val:.3f}."
            return HTMLResponse(_studio_shell(q, explanation, error="-", fig_json=fig, cyto=None, mermaid=None, segments_obj=None))

        # -------- Charts & Functions --------
        # 1) story-style walkthrough
        if intent == "chart":
            try:
                walk = get_chart_walkthrough(q, MODEL)
                if isinstance(walk, dict) and walk.get("segments"):
                    return HTMLResponse(_studio_shell(
                        q=q, explanation="Walkthrough with interleaved explanations and charts.",
                        error="-", fig_json=None, cyto=None, mermaid=None, segments_obj=walk
                    ))
            except Exception:
                pass

        # 2) function fallback
        if _looks_like_function(q):
            rng = _parse_range(q) or (-5.0, 5.0)
            fig = _sample_function(q, xmin=rng[0], xmax=rng[1], n=600)
            explanation = "Function plotted with a safe sampler. Adjust range below."
            return HTMLResponse(_studio_shell(q, explanation, error="-", fig_json=fig, cyto=None, mermaid=None, segments_obj=None))

        # 3) inline bar/pie parsing
        labels, vals = _parse_pairs(q)
        qs = (q or "").lower()
        if labels and vals:
            if "pie" in qs:
                if any(v < 0 for v in vals):
                    explanation = "Blocked: pie slices can’t be negative."
                    return HTMLResponse(_studio_shell(q, explanation, error="-", fig_json=None, cyto=None, mermaid=None, segments_obj=None))
                fig = {"data":[{"type":"pie","labels":labels,"values":vals}], "layout":{"title":"Pie chart"}}
                try: verify_pie({"labels": labels, "values": vals})
                except Exception: pass
                return HTMLResponse(_studio_shell(q, "Pie chart rendered from inline values.", error="-", fig_json=fig, cyto=None, mermaid=None, segments_obj=None))
            if "bar" in qs:
                fig = {"data":[{"type":"bar","x":labels,"y":vals}], "layout":{"title":"Bar chart","yaxis":{"title":"Value"}}}
                try: verify_bar({"x": labels, "y": vals})
                except Exception: pass
                return HTMLResponse(_studio_shell(q, "Bar chart rendered from inline values.", error="-", fig_json=fig, cyto=None, mermaid=None, segments_obj=None))

        # 4) simple line series
        if "line" in qs or "weekly" in qs:
            seq = re.findall(r"(\d+(?:\.\d+)?)", q)
            if seq:
                ys = [float(x) for x in seq]
                xs = list(range(1, len(ys)+1))
                fig = {"data":[{"type":"scatter","mode":"lines","x":xs,"y":ys}], "layout":{"title":"Line chart"}}
                try: verify_line({"x": xs, "y": ys})
                except Exception: pass
                return HTMLResponse(_studio_shell(q, "Line chart rendered from your series.", error="-", fig_json=fig, cyto=None, mermaid=None, segments_obj=None))

        # 5) last resort: chart spec via LLM
        try:
            raw = get_chart_spec_from_text(q, MODEL)
            if isinstance(raw, dict) and raw:
                if raw.get("mode") == "diagram":
                    mermaid_src = _sanitize_mermaid(raw.get("mermaid") or _default_mermaid())
                    walk = {
                        "segments": [
                            {"type": "text", "content": "Here’s a simple view of the scenario."},
                            {"type": "viz", "mode": "diagram", "mermaid": mermaid_src},
                            {"type": "text", "content": "Follow the arrows from left to right; controls attach at the critical steps."},
                        ],
                        "labels": []
                    }
                    return HTMLResponse(_studio_shell(q, "Concept diagram generated from your question.", error="-", fig_json=None, cyto=None, mermaid=None, segments_obj=walk))
                fig = _chart_spec_to_plotly(raw)
                explanation = raw.get("alt_text") or "Chart rendered from LLM spec."
                return HTMLResponse(_studio_shell(q, explanation, error="-", fig_json=fig, cyto=None, mermaid=None, segments_obj=None))
        except Exception:
            pass

        # default help
        explanation = ("Try a process diagram (e.g., phishing lifecycle), a function (e.g., “draw x^2”), "
                       "a ROC request (“positives ...; negatives ...”), or inline values for bar/pie/line. "
                       "You can also use the topic presets.")
        return HTMLResponse(_studio_shell(q, explanation, error="-", fig_json=None, cyto=None, mermaid=None, segments_obj=None))

    except Exception as e:
        explanation = "An error occurred."
        error = str(e)
        return HTMLResponse(_studio_shell(q, explanation, error, None, None, mermaid=None, segments_obj=None))

# ------------------------------ Shell renderer ------------------------------
def _studio_shell(
    q: str,
    explanation: str,
    error: str,
    fig_json,
    cyto,
    mermaid: str | None = None,
    segments_obj: dict | None = None,
) -> str:
    q_safe           = html.escape(q or "")
    explanation_safe = html.escape(_to_text(explanation) or "")
    error_safe       = html.escape(_to_text(error) or "-")
    fig_literal      = json.dumps(fig_json)  # None -> "null"
    cyto_json        = json.dumps(cyto or {}, ensure_ascii=False)
    mermaid_json     = json.dumps(mermaid) if mermaid else "null"
    preset_json      = json.dumps(TOPIC_PRESETS, ensure_ascii=False)

    # ---- Segments branch (LLM walkthrough) ----
    if segments_obj:
        segments = segments_obj.get("segments", [])
        blocks = []
        viz_payloads = []  # (kind, payload_json, container_id)
        viz_idx = 0
        first_mermaid_payload = None

        for seg in segments:
            t = (seg or {}).get("type")
            if t == "text":
                content = seg.get("content", "")
                blocks.append(f'<div class="seg seg-text">{content}</div>')
            elif t == "viz":
                mode = (seg.get("mode") or "").lower()
                cid = f"viz_{viz_idx}"; viz_idx += 1
                if mode in ("chart", "graph"):
                    blocks.append(f'<div id="{cid}" class="seg seg-viz" style="min-height:120px;margin:12px 0;"></div>')
                elif mode == "diagram":
                    blocks.append(
                        f'<div id="{cid}" class="seg seg-viz" '
                        'style="min-height:140px;margin:16px 0;border-bottom:1px solid #e5e7eb;padding-bottom:12px"></div>'
                    )
                if mode == "chart":
                    fig = _chart_spec_to_plotly(seg.get("chart") or {})
                    viz_payloads.append(("plotly", json.dumps(fig), cid))
                elif mode == "graph":
                    kg = seg.get("kg") or {}
                    try:
                        verify_kg(KnowledgeGraphSpec(**kg))
                        cyto_payload = kg_cyto_payload(kg)
                    except Exception:
                        cyto_payload = {"elements": {"nodes": [], "edges": []}}
                    viz_payloads.append(("cyto", json.dumps(cyto_payload, ensure_ascii=False), cid))
                elif mode == "diagram":
                    mer = seg.get("mermaid") or _default_mermaid()
                    payload = json.dumps(mer)
                    if first_mermaid_payload is None:
                        first_mermaid_payload = payload
                    viz_payloads.append(("mermaid", payload, cid))

        # Stepper, Ask panel, and Learning Lab
        stepper_html = """
<div id="stepper" style="display:flex;gap:8px;margin:12px 0;align-items:center">
  <button id="prev" type="button">Back</button>
  <span id="pos" style="min-width:60px;text-align:center"></span>
  <button id="next" type="button">Next</button>
</div>
"""
        ask_html = """
<div id="ask_panel" style="margin:12px 0 16px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
  <input id="ask_box" type="text" placeholder="Ask about this diagram (e.g., what to improve?)" style="flex:1;min-width:240px;padding:8px;border:1px solid #e5e7eb;border-radius:8px">
  <button id="ask_btn" type="button">Ask</button>
  <span id="ask_status" style="color:#6b7280"></span>
</div>
"""
        lab_html = """
<div id="lab" style="margin-top:20px;padding:16px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa">
  <h2 style="margin-top:0">Learning Lab</h2>
  <p style="color:#6b7280">Edit the diagram spec (nodes/edges/defenses). Quick-validate locally or run server notebook.</p>
  <label for="lab_spec" style="font-weight:600">Diagram spec JSON</label>
  <textarea id="lab_spec" style="width:100%;height:180px;margin-top:6px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;border:1px solid #e5e7eb;border-radius:8px;padding:8px"></textarea>
  <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
    <button id="btn_local_validate" type="button">Quick Validate (Local)</button>
    <button id="btn_validate" type="button">Validate in Notebook</button>
    <input id="lab_q" type="text" placeholder="Optional: what should I notice?" style="flex:1;min-width:220px;padding:8px;border:1px solid #e5e7eb;border-radius:8px"/>
    <button id="btn_explain" type="button">Explain Artifacts</button>
  </div>
  <div id="lab_out" style="margin-top:12px;background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px;white-space:pre-wrap"></div>
</div>
""".strip()

        # body
        body_html = stepper_html + ask_html + "\n".join(blocks) + ("\n" + lab_html if SHOW_LAB else "")
        js_lines = []

        # Viz renderers
        for kind, payload, cid in viz_payloads:
            if kind == "plotly":
                js_lines.append(
                    f"Plotly.newPlot('{cid}', ({payload}).data || ({payload})['data'], ({payload}).layout || ({payload})['layout'], {{responsive:true,displaylogo:false}});"
                )
            elif kind == "cyto":
                js_lines.append(
                    f"""
cytoscape({{
  container: document.getElementById('{cid}'),
  elements: ({payload}).elements || [],
  layout: {{ name: 'cose', animate: false }},
  style: [
    {{
      selector: 'node',
      style: {{
        'label':'data(label)',
        'background-color':'#3b82f6',
        'color':'#111827',
        'font-size':12
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'curve-style':'bezier',
        'target-arrow-shape':'triangle',
        'line-color':'#9ca3af',
        'target-arrow-color':'#9ca3af',
        'label':'data(rel)',
        'font-size':10
      }}
    }}
  ]
}});
"""
                )
            elif kind == "mermaid":
                js_lines.append(
                    f"""
mermaid.render('m_{cid}', {payload}).then(({{svg, bindFunctions}}) => {{
  document.getElementById('{cid}').innerHTML = svg;
  if (bindFunctions) bindFunctions(document.getElementById('{cid}'));
}}).catch(err => {{
  document.getElementById('{cid}').innerHTML = '<div style="color:#b91c1c">Mermaid error: ' + (err && err.message ? err.message : err) + '</div>';
}});
"""
                )

        # Stepper/pulse + Preset dropdown + Local/Server validators + Ask + localStorage
        js_lines.append(f"""
(function uiInit(){{
  try {{
    const form = document.querySelector('form[action="/studio"]');
    if (form) {{
      const sel = document.createElement('select');
      sel.id = 'preset';
      sel.style.cssText = 'padding:10px;border:1px solid #d1d5db;border-radius:8px;margin-right:8px';
      sel.innerHTML = '<option value="">Topic presets…</option>' + Object.keys({preset_json}).map(k => '<option>'+k+'</option>').join('');
      const qbox = form.querySelector('input[name="q"]');
      form.insertBefore(sel, qbox);
      sel.onchange = () => {{
        const k = sel.value;
        if (k && {preset_json}[k]) qbox.value = {preset_json}[k];
      }};
    }}
  }} catch(e){{}}

  const blocks = Array.from(document.querySelectorAll('.seg'));
  let i = 0;
  function show(k){{
    i = Math.max(0, Math.min(blocks.length-1, k));
    blocks.forEach((b,idx)=> b.style.display = (idx===i ? '' : 'none'));
    const pos = document.getElementById('pos'); if (pos) pos.textContent = (i+1)+'/'+blocks.length;
    window.dispatchEvent(new CustomEvent('tv-step-change', {{detail:{{index:i}}}}));
  }}
  const prev = document.getElementById('prev');
  const next = document.getElementById('next');
  if (prev) prev.onclick = ()=>show(i-1);
  if (next) next.onclick = ()=>show(i+1);
  if (blocks.length) show(0);

  const style = document.createElement('style');
  style.textContent = '.pulse{{animation:pulse .9s ease-out 1}}@keyframes pulse{{0%{{filter:drop-shadow(0 0 0 #34d399)}}50%{{filter:drop-shadow(0 0 8px #34d399)}}100%{{filter:drop-shadow(0 0 0 #34d399)}}}}';
  document.head.appendChild(style);
  window.addEventListener('tv-step-change', (ev) => {{
    const seg = blocks[ev.detail.index];
    if (!seg) return;
    const rect = seg.querySelector('svg g.node rect');
    if (rect) {{ rect.classList.add('pulse'); setTimeout(()=>rect.classList.remove('pulse'), 900); }}
  }});

  // Learning Lab bootstrap (prefill spec) + localStorage persistence
  try {{
    const ta = document.getElementById('lab_spec');
    if (ta) {{
      const KEY = 'tv_last_spec';
      const saved = localStorage.getItem(KEY);
      let spec = saved ? JSON.parse(saved) : {{"nodes":[{{"id":"A","label":"Recon"}},{{"id":"B","label":"Initial Access"}}],
                   "edges":[["A","B",{{"label":"progression"}}]],
                   "defenses":[{{"attachTo":"B","label":"MFA"}}]}};
      ta.value = JSON.stringify(spec, null, 2);
      ta.addEventListener('input', () => {{
        try {{ localStorage.setItem(KEY, ta.value); }} catch(_){{
          /* ignore quota errors */
        }}
      }});
    }}
  }} catch(e){{}}

  // Local validator
  const btnLocal = document.getElementById('btn_local_validate');
  if (btnLocal) btnLocal.onclick = () => {{
    const out = document.getElementById('lab_out');
    try {{
      const spec = JSON.parse(document.getElementById('lab_spec').value);
      const nodes = new Set((spec.nodes||[]).map(n => n.id || n.label).filter(Boolean));
      const edges = (spec.edges||[]).map(e => Array.isArray(e) ? [e[0],e[1]] : [e.src||e.source, e.dst||e.target]);
      const deg = {{}}; nodes.forEach(n=>deg[n]=0);
      edges.forEach(([a,b])=>{{ if(a&&b){{ deg[a]=(deg[a]||0)+1; deg[b]=(deg[b]||0)+1; }} }});
      const isolated = Object.entries(deg).filter(([,d]) => d===0).map(([n])=>n);
      const adj = {{}}; nodes.forEach(n=>adj[n]=[]);
      edges.forEach(([a,b])=>{{ if(a&&b) adj[a].push(b); }});
      function hasCycle(){{
        const vis = new Set(), stack = new Set();
        function dfs(v){{
          vis.add(v); stack.add(v);
          for(const w of adj[v]){{ if(!vis.has(w) && dfs(w)) return true; if(stack.has(w)) return true; }}
          stack.delete(v); return false;
        }}
        for(const n of nodes){{ if(!vis.has(n) && dfs(n)) return true; }}
        return false;
      }}
      const verdicts = [];
      if (hasCycle()) verdicts.push("Flow contains cycles; remove back-edges.");
      if ((spec.defenses||[]).length === 0) verdicts.push("No defensive controls attached.");
      if (isolated.length) verdicts.push("Isolated nodes: " + JSON.stringify(isolated.slice(0,5)));
      const result = {{checks:{{node_count:nodes.size, edge_count:edges.length, isolated_nodes:isolated}}, verdicts: verdicts.length?verdicts:["OK"]}};
      out.textContent = "Artifacts (local):\\n" + JSON.stringify(result, null, 2);
      window.__tv_artifacts = result;
    }} catch(e){{
      out.textContent = "Invalid spec JSON.";
    }}
  }};

  // Server-side notebook validate
  const btnValidate = document.getElementById('btn_validate');
  if (btnValidate) btnValidate.onclick = async () => {{
    const out = document.getElementById('lab_out');
    out.textContent = 'Running notebook on server…';
    try {{
      const spec = JSON.parse(document.getElementById('lab_spec').value);
      const res = await fetch('/lab/validate', {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(spec)
      }});
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'validate failed');
      window.__tv_artifacts = data.artifacts;
      out.textContent = "Artifacts (server):\\n" + JSON.stringify(data.artifacts, null, 2);
    }} catch (e) {{
      out.textContent = "Server validation error: " + (e && e.message ? e.message : e);
    }}
  }};

  // LLM explanation of artifacts
  const btnExplain = document.getElementById('btn_explain');
  if (btnExplain) btnExplain.onclick = async () => {{
    const out = document.getElementById('lab_out');
    out.textContent = 'Asking the model to explain notebook results…';
    try {{
      const spec = JSON.parse(document.getElementById('lab_spec').value);
      const q    = (document.getElementById('lab_q')?.value || '').trim();
      const res = await fetch('/lab/explain', {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{ spec, q }})
      }});
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'explain failed');
      const pre = "Artifacts (server):\\n" + JSON.stringify(data.artifacts, null, 2) + "\\n\\n";
      out.textContent = pre + (data.explanation || "(no explanation returned)");
    }} catch (e) {{
      out.textContent = "Explain error: " + (e && e.message ? e.message : e);
    }}
  }};

  // Optional: Ask about this diagram
  const askBtn = document.getElementById('ask_btn');
  const askBox = document.getElementById('ask_box');
  const askStatus = document.getElementById('ask_status');
  if (askBtn && askBox) {{
    // expose first diagram mermaid to window for /studio/ask
    window.__tv_first_mermaid = {first_mermaid_payload if first_mermaid_payload else "null"};
    askBtn.onclick = async () => {{
      const q = (askBox.value || '').trim();
      if (!q) {{ askStatus.textContent = 'Type a question first.'; return; }}
      askStatus.textContent = 'Thinking…';
      try {{
        const res = await fetch('/studio/ask', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ mermaid: window.__tv_first_mermaid || '', q }})
        }});
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'ask failed');
        askStatus.textContent = '';
        alert(data.answer || '(no answer)');
      }} catch (e) {{
        askStatus.textContent = 'Ask error: ' + (e && e.message ? e.message : e);
      }}
    }};
  }}
}})();
""")

        js_all = "\n".join(js_lines)

        return f"""
<html>
  <head>
    <meta charset="utf-8"/>
    <title>TrustViz Studio</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px }}
      form {{ display:flex; gap:8px; align-items:center }}
      input[type=text] {{ flex:1; padding:10px; border:1px solid #d1d5db; border-radius:8px }}
      button {{ padding:10px 16px }}
      .seg-text {{ line-height:1.55; font-size:15px }}
    </style>
  </head>
  <body>
    <h1>TrustViz Studio</h1>
    <form action="/studio" method="get">
      <input type="text" name="q" value="{q_safe}" placeholder="Ask or choose a preset…"/>
      <button type="submit">Ask</button>
    </form>

    <h2>Explanation</h2>
    <p>{explanation_safe}</p>
    <p style="color:#6b7280">Error: {error_safe}</p>

    {body_html}

    <script>
      mermaid.initialize({{ startOnLoad: false, securityLevel: "strict", theme: "default" }});
      {js_all}
    </script>
  </body>
</html>
"""

    # ---- Mermaid-only ----
    if mermaid:
        return f"""
<html>
  <head>
    <meta charset="utf-8"/>
    <title>TrustViz Studio</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px }}
      form {{ display:flex; gap:8px; align-items:center }}
      input[type=text] {{ flex:1; padding:10px; border:1px solid #d1d5db; border-radius:8px }}
      button {{ padding:10px 16px }}
      #m1 {{ border:1px solid #e5e7eb; border-radius:12px; margin-top:12px; padding:12px; background:#fff }}
    </style>
  </head>
  <body>
    <h1>TrustViz Studio</h1>
    <form action="/studio" method="get">
      <input type="text" name="q" value="{q_safe}" placeholder="Ask or choose a preset…"/>
      <button type="submit">Ask</button>
    </form>

    <h2>Explanation</h2>
    <p>{explanation_safe}</p>
    <p style="color:#6b7280">Error: {error_safe}</p>

    <div id="m1" aria-label="Concept diagram"></div>

    <script>
      mermaid.initialize({{ startOnLoad: false, securityLevel: "strict", theme: "default" }});
      const src = {mermaid_json};
      mermaid.render("graph1", src).then(({{svg, bindFunctions}}) => {{
        const el = document.getElementById("m1");
        el.innerHTML = svg;
        if (bindFunctions) bindFunctions(el);
      }}).catch(err => {{
        const el = document.getElementById("m1");
        el.innerHTML = '<div style="padding:16px;color:#b91c1c">Mermaid error: ' +
                       (err && err.message ? err.message : String(err)) + '</div>';
      }});
    </script>
  </body>
</html>
"""

    # ---- Cytoscape branch ----
    if cyto is not None:
        return f"""
<html>
  <head>
    <meta charset="utf-8"/>
    <title>TrustViz Studio</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px }}
      form {{ display:flex; gap:8px; align-items:center }}
      input[type=text] {{ flex:1; padding:10px; border:1px solid #d1d5db; border-radius:8px }}
      button {{ padding:10px 16px }}
      #cy {{ height:560px; border:1px solid #e5e7eb; border-radius:12px; margin-top:12px; background:#fff }}
    </style>
  </head>
  <body>
    <h1>TrustViz Studio</h1>
    <form action="/studio" method="get">
      <input type="text" name="q" value="{q_safe}" placeholder="Describe a scenario to build a graph…"/>
      <button type="submit">Ask</button>
    </form>

    <h2>Explanation</h2>
    <p>{explanation_safe}</p>
    <p style="color:#6b7280">Error: {error_safe}</p>

    <div id="cy" aria-label="Knowledge graph"></div>

    <script>
      const payload = {cyto_json};
      cytoscape({{
        container: document.getElementById('cy'),
        elements: payload.elements || [],
        layout: {{ name: 'cose', animate: false }},
        style: [
          {{
            selector: 'node',
            style: {{
              'label': 'data(label)',
              'background-color': '#3b82f6',
              'color': '#111827',
              'text-valign': 'center',
              'text-halign': 'center',
              'font-size': 12,
              'width': 30,
              'height': 30
            }}
          }},
          {{
            selector: 'edge',
            style: {{
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'line-color': '#9ca3af',
              'target-arrow-color': '#9ca3af',
              'label': 'data(rel)',
              'font-size': 10,
              'text-background-color': '#ffffff',
              'text-background-opacity': 0.85,
              'text-background-padding': 2
            }}
          }}
        ]
      }});
    </script>
  </body>
</html>
"""

    # ---- Plotly (charts / math) branch ----
    return f"""
<html>
  <head>
    <meta charset="utf-8"/>
    <title>TrustViz Studio</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px }}
      form {{ display:flex; gap:8px; align-items:center }}
      input[type=text] {{ flex:1; padding:10px; border:1px solid #d1d5db; border-radius:8px }}
      button {{ padding:10px 16px }}
      #chart {{ height:520px; border:1px solid #e5e7eb; border-radius:12px; margin-top:12px; background:#fff }}
      .controls {{ margin:8px 0; display:flex; gap:8px; align-items:center }}
      .controls input[type=number] {{ width:120px; padding:6px; border:1px solid #e5e7eb; border-radius:8px }}
    </style>
  </head>
  <body>
    <h1>TrustViz Studio</h1>
    <form action="/studio" method="get">
      <input type="text" name="q" value="{q_safe}" placeholder="Ask for a function (e.g., draw x^2), ROC, or give bar/pie/line values…"/>
      <button type="submit">Ask</button>
    </form>

    <h2>Explanation</h2>
    <p>{explanation_safe}</p>
    <p style="color:#6b7280">Error: {error_safe}</p>

    <div class="controls" aria-label="Chart controls">
      <label>xmin <input id="xmin" type="number" step="0.5" /></label>
      <label>xmax <input id="xmax" type="number" step="0.5" /></label>
      <button id="rescale" type="button">Rescale</button>
    </div>

    <div id="chart" role="img" aria-label="Chart"></div>

    <script>
      const rawFig = {fig_literal};
      const fig = (typeof rawFig === 'string') ? (rawFig ? JSON.parse(rawFig) : null) : rawFig;

      if (fig) {{
        fig.layout = fig.layout || {{}};
        fig.layout.dragmode = 'pan';
        fig.layout.hovermode = 'x unified';
        fig.layout.margin = {{l:60, r:20, t:40, b:60}};
        fig.layout.xaxis = Object.assign({{
          title: {{text: (fig.layout.xaxis && fig.layout.xaxis.title && fig.layout.xaxis.title.text) || 'x'}},
          zeroline: true, zerolinewidth: 2,
          showline: true, linewidth: 1, mirror: true,
          gridcolor: '#e5e7eb',
          showspikes: true, spikemode: 'across', spikesnap: 'cursor'
        }}, fig.layout.xaxis || {{}});
        fig.layout.yaxis = Object.assign({{
          title: {{text: (fig.layout.yaxis && fig.layout.yaxis.title && fig.layout.yaxis.title.text) || 'y'}},
          zeroline: true, zerolinewidth: 2,
          showline: true, linewidth: 1, mirror: true,
          gridcolor: '#e5e7eb',
          showspikes: true, spikemode: 'across', spikesnap: 'cursor'
        }}, fig.layout.yaxis || {{}});

        (fig.data || []).forEach(tr => {{
          if (!tr.mode) tr.mode = 'lines';
          tr.line = Object.assign({{width: 3}}, tr.line || {{}});
        }});

        const config = {{
          responsive: true,
          displaylogo: false,
          scrollZoom: true,
          modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d','toggleSpikelines']
        }};

        Plotly.newPlot('chart', fig.data, fig.layout, config);

        const xIn = document.getElementById('xmin');
        const xAx = document.getElementById('xmax');
        const btn = document.getElementById('rescale');
        if (fig.data && fig.data.length && fig.data[0].x && xIn && xAx && btn) {{
          const xs = (fig.data[0].x || []).map(Number).filter(v => Number.isFinite(v));
          if (xs.length) {{
            xIn.value = Math.min(...xs).toFixed(2);
            xAx.value = Math.max(...xs).toFixed(2);
          }}
          btn.onclick = () => {{
            const lo = parseFloat(xIn.value);
            const hi = parseFloat(xAx.value);
            if (isFinite(lo) && isFinite(hi) && lo < hi) {{
              Plotly.relayout('chart', {{'xaxis.range': [lo, hi]}});
            }}
          }};
        }}
      }}
    </script>
  </body>
</html>
"""

