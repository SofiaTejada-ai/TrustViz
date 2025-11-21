# trustviz/server/studio_routes.py — ScholarViz (LLM→Mermaid + sources + robust OA + graceful grounded fallback)
# + Diversity knobs: n, seed, diversify, temp (controlled randomness for grounded variants)
# + Export endpoint (SVG/PNG) with optional light branding (does not alter LLM/grounded flow)
# + FIX: preserve subgraph/style/class lines and hex colors so yellow “Controls” box renders

# trustviz/server/studio_routes.py — ScholarViz (LLM→Mermaid + sources + robust OA + grounded fallback)
# + Diversity knobs: n, seed, diversify, temp
# + Export endpoint (SVG/PNG) with optional light branding
# + FIX: preserve subgraph/style/class lines and hex colors

from __future__ import annotations

import os, re, io, json, html, base64, datetime, random
from typing import List, Tuple

from fastapi import APIRouter, Body, UploadFile, File, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pypdf import PdfReader
import httpx

from trustviz.llm.gemini_client import gen_text, gen_json
from trustviz.notebook.runner import NotebookRunner

router = APIRouter(tags=["scholarviz"])
MODEL = os.environ.get("TRUSTVIZ_LLM_MODEL", "gemini-2.5-pro")

# …rest of your ScholarViz endpoints here (NO include_router calls) …


# ---------- safety ----------
_DENY = [
  r"\b(ddos|dos)\b", r"\b(botnet|c2|command\s*and\s*control)\b", r"\b(keylogger|rootkit|rat)\b",
  r"\bzero[-\s]?day\b", r"\bexploit\s+code\b", r"\bcredential\s+stuffing\b", r"\bphishing\s+kit\b",
  r"\bpassword\s*cracker\b", r"\bport\s*scan\s*(script|code)\b", r"\bsql\s*injection\s*(payload|exploit)\b",
  r"\bbypass\s+mfa\b", r"\breal\s+bank\s+site\b", r"\bhow\s+to\s+(hack|bypass)\b", r"\bbypass\b", r"\bhack(?:ing)?\b"
]
USE_GROUNDED_DEFAULT = True

def _risky(s:str)->bool: return any(re.search(p,(s or "").lower()) for p in _DENY)

# ---------- mermaid allowlist/sanitize ----------
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]()_:.-> \n|/%'\",+&?-#;")
def _sanitize_mermaid(s:str)->str:
  return "".join(ch for ch in (s or "") if ch in _ALLOWED)

_NODE_DEF = re.compile(r"^\s*([A-Za-z0-9_]+)\s*[\[\(]", re.M)
_EDGE = re.compile(r"\b([A-Za-z0-9_]+)\s*-[.-]*>\s*([A-Za-z0-9_]+)\b")

def _ensure_node_defs(s: str) -> str:
    if not s.strip(): return s
    existing = set(m.group(1) for m in _NODE_DEF.finditer(s))
    refs = set()
    for a, b in _EDGE.findall(s):
        refs.add(a); refs.add(b)
    missing = [nid for nid in refs if nid not in existing]
    if not missing: return s
    lines = s.splitlines()
    insert_at = 1 if lines else 0
    for nid in missing:
        lines.insert(insert_at, f"{nid}[{nid}]")
        insert_at += 1
    return "\n".join(lines)

def _default_mermaid()->str:
  return ("flowchart LR\n"
          "  Q[Question] --> D[High-level Diagram]\n"
          "  D --> K[Key Controls]\n"
          "  K --> O[Outcomes]\n")

# ---------- OA query normalization ----------
def _normalize_for_oa(q: str) -> List[str]:
  if not q: return []
  q0 = re.sub(r"[“”‘’\"'’→⇒—–\-–>]+", " ", q)
  q0 = re.sub(r"[^A-Za-z0-9\s\+\.]", " ", q0)
  q0 = re.sub(r"\s+", " ", q0).strip()

  lower = q0.lower()
  expanded = lower
  expanded = expanded.replace("dlp", "data loss prevention")
  expanded = expanded.replace("saas", "software as a service")
  expanded = expanded.replace("zero trust", "zero trust architecture")

  tiers = []
  tiers.append(expanded)
  if "data loss prevention" in expanded or "dlp" in lower:
    tiers.append("data loss prevention policy classification quarantine allow cloud")
  if "software as a service" in expanded or "saas" in lower:
    tiers.append("cloud security SaaS policy engine content inspection classification")
  tiers.append(re.sub(r"\b(users?|apps?|classifiers?|policy|quarantine|allow)\b", "", expanded).strip())

  out = []
  for t in tiers:
    t = re.sub(r"\s+", " ", t).strip()
    if t and t not in out: out.append(t)
  return [o for o in out if o]

# ---------- sources (OpenAlex) ----------
async def _scholar_search(q:str, per:int=5)->List[dict]:
  now = datetime.date.today()
  start = f"{now.year-3}-01-01"
  url = "https://api.openalex.org/works"
  params = {
    "search": q,
    "filter": f"from_publication_date:{start},language:en,primary_location.source.type:journal",
    "sort": "publication_date:desc",
    "per_page": per
  }
  try:
    async with httpx.AsyncClient(timeout=12.0) as hx:
      r = await hx.get(url, params=params)
      r.raise_for_status()
      data = r.json().get("results", [])
  except Exception:
    return []
  out = []
  for w in data:
    out.append({
      "title": w.get("title") or "(untitled)",
      "year": (w.get("publication_year") or ""),
      "venue": (w.get("primary_location",{}).get("source",{}) or {}).get("display_name"),
      "url": (w.get("primary_location",{}) or {}).get("landing_page_url") or w.get("open_access",{}).get("oa_url") or w.get("id"),
      "authors": ", ".join(a.get("author",{}).get("display_name","") for a in (w.get("authorships") or [])[:4])
    })
  return out

def _labels_from_mermaid(src:str)->List[str]:
  labs, seen = [], set()
  for m in re.finditer(r"\b[A-Za-z0-9_]+\s*[\[\(]([^\]\)]+)[\]\)]", src or ""):
    t = m.group(1).strip()
    if t and t not in seen: seen.add(t); labs.append(t)
  return labs[:20]

# ---------- special diagrams (deterministic fallbacks) ----------
def _diagram_cia()->str:
  return _sanitize_mermaid("""flowchart LR
C[Confidentiality] --> P1[Access Control]
I[Integrity] --> P2[Checksums/Signing]
A[Availability] --> P3[Redundancy/Backups]
C -.-> DC1[Encryption at Rest/In Transit]
I -.-> DC2[Hash, Code Signing, WORM]
A -.-> DC3[Failover, DDoS Mitigation]
""")

def _diagram_zerotrust()->str:
  return _sanitize_mermaid("""flowchart LR
R[Request] --> P[Policy: Identity + Device + Context]
P --> G[Grant Least Privilege]
G --> V[Continuous Verification]
V --> Rv[Re-evaluate/Block]
P -.-> M1[MFA/WebAuthn]
P -.-> M2[Device Posture]
G -.-> M3[Segmentation]
""")

def _diagram_dlp()->str:
  return _sanitize_mermaid("""flowchart LR
U[Users] --> A[SaaS Apps]
A --> C[Content Classifiers]
C --> Po[Policy Engine]
Po --> Ac[Quarantine/Allow]
Po -.-> JIT[Just-in-Time Exceptions]
Po -.-> Aud[Audit / Alerts]
C -.-> Enc[Encryption at Rest/In Transit]
""")

# ---------- ASCII ("A -> B : label") to Mermaid ----------
_ASCII_ARROW_LINE = re.compile(r"^\s*([^#\-\*].*?)\s*(-{1,2}|=|–|—)?>\s*(.+)$")
_ARROWS = ("->", "-->", "=>", "→", "⇒", "—>", "–>")

def _slug(s: str) -> str:
  s = re.sub(r"[^A-Za-z0-9_]+", "_", s.strip())
  return (s or "n").strip("_")[:40] or "n"

def _extract_flow_lines(s: str) -> List[Tuple[str, str, str]]:
  flows: List[Tuple[str, str, str]] = []
  if not s: return flows
  for raw in s.splitlines():
    line = raw.strip()
    if not line: continue
    if any(tok in line for tok in _ARROWS):
      parts = re.split(r"\s*(?:-+>|=>|→|⇒|—>)\s*", line, maxsplit=1)
      if len(parts) == 2:
        a, rest = parts[0].strip(), parts[1].strip()
        if ":" in rest:
          b, lbl = rest.split(":", 1)
          flows.append((a.strip(), b.strip(), lbl.strip()))
        else:
          flows.append((a.strip(), rest.strip(), ""))
        continue
    m = _ASCII_ARROW_LINE.match(line)
    if m:
      a, _, b = m.groups()
      if ":" in b:
        b2, lbl = b.split(":", 1)
        flows.append((a.strip(), b2.strip(), lbl.strip()))
      else:
        flows.append((a.strip(), b.strip(), ""))
  return flows

def _ascii_to_mermaid(s: str, limit: int = 8) -> str:
  flows = _extract_flow_lines(s)
  if not flows: return ""
  nodes: dict = {}
  edges: List[Tuple[str, str, str]] = []
  for a, b, lbl in flows:
    if not a or not b: continue
    if len(nodes) < limit: nodes.setdefault(a, _slug(a))
    if len(nodes) < limit: nodes.setdefault(b, _slug(b))
    if a in nodes and b in nodes: edges.append((a, b, lbl))
  if len(nodes) < 2 or not edges: return ""
  lines = ["flowchart LR"]
  for human, nid in nodes.items(): lines.append(f"{nid}[{human}]")
  for a, b, lbl in edges:
    aa, bb = nodes[a], nodes[b]
    lines.append(f"{aa} -- {lbl} --> {bb}" if lbl else f"{aa} --> {bb}")
  return _sanitize_mermaid("\n".join(lines))

# ---------- Mermaid repair / normalization ----------
_MER_BLOCK = re.compile(r"```(?:mermaid)?\s*([\s\S]*?)```", re.I)

def _strip_fences(s:str)->str:
  m = _MER_BLOCK.search(s or "")
  return m.group(1).strip() if m else (s or "")

def _coerce_flowchart_lr(s: str) -> str:
  s = (s or "").strip()
  if not s:
    return "flowchart LR"
  lines = s.splitlines()
  first = lines[0].strip()
  m = re.match(r"(?i)^\s*(graph|flowchart)\s+(LR|RL|TB|BT)\b(.*)$", first)
  if m:
    rest = (m.group(3) or "").strip()
    new_lines = ["flowchart LR"]
    if rest:
      new_lines.append(rest)
    new_lines.extend(lines[1:])
    return "\n".join(new_lines)
  if not re.match(r"(?i)^\s*(graph|flowchart)\b", first):
    return "flowchart LR\n" + s
  lines[0] = "flowchart LR"
  return "\n".join(lines)

def _normalize_arrows(s: str) -> str:
  return (s or "").replace("–>", "->").replace("—>", "->").replace("⇒", "->").replace("→", "->")
def _strip_inline_headers(s: str) -> str:
  def _clean_line(i, line):
    if i == 0:
      return line
    line = re.sub(r"(?i)\bgraph\s+(LR|RL|TB|BT)\b", "", line)
    line = re.sub(r"(?i)\bflowchart\s+(LR|RL|TB|BT)\b", "", line)
    return re.sub(r"\s{2,}", " ", line).strip()

  lines = (s or "").splitlines()
  return "\n".join(_clean_line(i, ln) for i, ln in enumerate(lines))

def _trim_extraneous_text(s: str) -> str:
  keep_prefixes = ("flowchart","graph","classDiagram","sequenceDiagram","stateDiagram",
                   "subgraph","end","classDef","style","linkStyle","click")
  out = []
  for line in (s.splitlines() if s else []):
    t = (line or "").strip()
    if not t:
      continue
    tl = t.lower()
    if tl.startswith(keep_prefixes):
      out.append(t)
      continue
    if "->" in t or "-.->" in t or re.search(r"\b[A-Za-z0-9_]+\s*[\[\(].*[\]\)]", t):
      out.append(t)
      continue
  return "\n".join(out).strip()

def _is_plausible_mermaid(s: str) -> bool:
  if not s or "flowchart" not in (s or "").lower():
    return False
  if any(k in s for k in ("subgraph", "classDef", "style", "linkStyle")):
    if re.search(r"\b[A-Za-z0-9_]+\s*[\[\(][^\]\)]+[\]\)]", s): return True
    if "->" in s or "-.->" in s: return True
  if "->" not in s and "-.->" not in s: return False
  nodes = re.findall(r"\b[A-Za-z0-9_]+\s*[\[\(][^\]\)]+[\]\)]", s)
  return len(nodes) >= 2

_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9+\-/ ]{1,40}")

def _pick_labels_from_text(t: str, k: int = 6) -> list[str]:
    t = (t or "").replace("\n", " ")
    toks = _WORDS.findall(t)
    bad = {"the","and","with","that","this","they","these","those","which",
           "into","from","over","under","using","based","since","because",
           "are","is","be","been","being","for","to","of","in","on","as"}
    freq = {}
    for w in toks:
        w2 = w.strip().strip("-/").strip()
        lw = w2.lower()
        if len(w2) < 3 or lw in bad:
            continue
        freq[w2] = freq.get(w2, 0) + 1
    scored = sorted(freq.items(), key=lambda kv: (-(kv[1] + 0.1*(1 if " " in kv[0] else 0)), -len(kv[0])))
    return [w for w, _ in scored[:k]]

def _skeleton_from_text(t: str) -> str:
    labs = _pick_labels_from_text(t, k=7)
    if len(labs) < 3:
        labs = (labs + ["Inputs","Processing","Outputs"])[:5]
    lines = ["flowchart LR"]
    ids = []
    for i, lab in enumerate(labs):
        nid = f"N{i+1}"
        ids.append(nid)
        lines.append(f'{nid}[{lab[:40]}]')
    for i in range(len(ids)-1):
        lines.append(f"{ids[i]} --> {ids[i+1]}")
    if len(ids) >= 4:
        lines.append(f"{ids[0]} -.-> {ids[2]}")
        lines.append(f"{ids[1]} -.-> {ids[3]}")
    return "\n".join(lines)

def _repair_mermaid(src: str) -> str:
  s = _strip_fences(src)
  s = _trim_extraneous_text(s)
  s = _coerce_flowchart_lr(s)
  s = _normalize_arrows(s)
  s = _strip_inline_headers(s)
  s = _sanitize_mermaid(s)

  has_subgraph = bool(re.search(r"(?i)^\s*subgraph\b", s, re.M))

  good = []
  for line in s.splitlines():
    t = line.strip()
    if not t:
      continue
    tl = t.lower()
    if tl.startswith(("flowchart","subgraph","end","classdef","style","linkstyle","click")):
      good.append(t)
      continue
    if t.count("[") == t.count("]") and t.count("(") == t.count(")"):
      good.append(t)

  s2 = "\n".join(good).strip()

  if not has_subgraph:
    s2 = _ensure_node_defs(s2)

  if "-.->" in s2 and "classDef dashed" not in s2:
    s2 += "\nclassDef dashed stroke-dasharray: 4 3,stroke:#888,color:#555;"

  if _is_plausible_mermaid(s2):
    return s2

  ascii_try = _ascii_to_mermaid(src)
  if _is_plausible_mermaid(ascii_try):
    if not has_subgraph:
      ascii_try = _ensure_node_defs(ascii_try)
    if "-.->" in ascii_try and "classDef dashed" not in ascii_try:
      ascii_try += "\nclassDef dashed stroke-dasharray: 4 3,stroke:#888,color:#555;"
    return ascii_try

  synth = _skeleton_from_text(src or "")
  return synth if _is_plausible_mermaid(synth) else ""

_FEWSHOT = [
  {
    "q": "Sketch Data Loss Prevention for SaaS",
    "mermaid": """flowchart LR
U[Users] --> A[SaaS Apps]
A --> C[Content Classifiers]
C --> P[Policy Engine]
P -->|Violation| Q[Quarantine]
P -->|No Violation| AL[Allow]
P -.-> JIT[Just-in-Time Exceptions]
P -.-> AUD[Audit/Alerts]"""
  },
  {
    "q": "Zero Trust access decision with device posture and MFA",
    "mermaid": """flowchart LR
REQ[Request] --> POL[Policy (Identity+Device+Context)]
POL -->|Grant| LEAST[Least Privilege]
LEAST --> MON[Continuous Verification]
POL -.-> MFA[MFA/WebAuthn]
POL -.-> POST[Device Posture]
MON -->|Re-evaluate| BLOCK[Block/Step-up]"""
  },
  {
    "q":"Incident Response lifecycle with playbooks and post-mortem",
    "mermaid": "flowchart LR\nDET[Detection] --> ANA[Analysis]\nANA --> CON[Containment]\nCON --> ERA[Eradication]\nERA --> REC[Recovery]\nREC --> PM[Post-Mortem]\nPB(Playbooks) -.-> DET\nPB -.-> ANA\nPB -.-> CON\nPB -.-> ERA\nPB -.-> REC"
  }
]

def _llm_mermaid_plan(q: str, req_labels: List[str] | None = None) -> Tuple[str, str]:
  req_labels = req_labels or []
  schema = (
    "Return JSON exactly as {make_diagram:boolean, mermaid:string, summary:string}. "
    "The 'mermaid' MUST be a valid Mermaid flowchart with 'flowchart LR' on the first line. "
    "You MAY group related nodes using 'subgraph <Title>' ... 'end' and style the group, e.g. "
    "'style <Title> fill:#FEF3C7,stroke:#EAB308,stroke-width:1px'. "
    "Do NOT use code fences. 5–8 nodes. Prefer nouns in square boxes. "
    "Use dashed edges for controls/side-constraints. "
    + ("Include these node titles verbatim: " + ", ".join(req_labels) + ". " if req_labels else "")
  )
  shots = "\n\n".join(f"Q: {s['q']}\nMermaid:\n{s['mermaid']}" for s in _FEWSHOT)
  prompt = f"{schema}\n\n{shots}"
  res = gen_json(prompt, q, model=MODEL) or {}
  raw_mer = (res.get("mermaid") or "").strip()
  mer = _repair_mermaid(raw_mer) or _ascii_to_mermaid(raw_mer) or ""
  return mer, (res.get("summary") or "")

ASK_RULES = "Defense-only; <=160 words; use exact diagram labels if helpful; no offensive steps."
DOC_RULES = (
  "Summarize like for a security student: 2-sentence overview; 3 key bullets; 2 defender takeaways. "
  "If a question is provided, answer briefly at the end."
)

_INGEST = {"text":"", "images":[]}

def _inject_brand_css(svg_text: str, brand: dict | None) -> str:
  if not brand:
    return svg_text
  primary = brand.get("primary", "#0ea5e9")
  stroke  = brand.get("stroke",  "#0f172a")
  font    = brand.get("font",    "ui-sans-serif, system-ui, -apple-system")
  rounded = brand.get("rounded", True)

  style = f"""
  <style>
    .mermaid text {{ font-family: {font}; fill: {stroke}; }}
    .mermaid .node rect, .mermaid .node polygon, .mermaid .cluster rect {{
      fill: {primary}1A; stroke: {stroke}; {'rx: 12px; ry: 12px;' if rounded else ''} 
    }}
    .mermaid .edgePath path {{ stroke: {stroke}; }}
    .mermaid .label text {{ fill: {stroke}; }}
    .mermaid .cluster text {{ fill: {stroke}; font-weight: 600; }}
    .mermaid .flowchart-link {{ stroke: {stroke}; }}
    .mermaid .edgePath path[stroke-dasharray] {{ stroke: {stroke}; }}
  </style>
  """.strip()

  return re.sub(r"(<svg[^>]*>)", r"\1" + style, svg_text, count=1, flags=re.I)

@router.post("/ask")
async def ask(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  if _risky(q): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  try:
    q_norm = _normalize_for_oa(q)[0] if _normalize_for_oa(q) else q
    sources = await _scholar_search(q_norm, per=5)
    a = gen_text("Answer for a cybersecurity audience. Cite sources by [#] inline where useful.", q, model=MODEL) or ""
    return JSONResponse({"ok":True, "answer":a.strip(), "sources":sources})
  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)

@router.post("/diagram/plan")
async def diagram_plan(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  if _risky(q): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  ql = q.lower()

  if USE_GROUNDED_DEFAULT:
    hits = await _open_access_hits(q, per=8)
    if hits:
      texts = [(h, await _fetch_pdf_text(h["pdf"])) for h in hits[:3]]
      pool = []
      for h, t in texts:
        if not t: continue
        pool.extend([f"[cap] {c}" for c in _fig_captions(t)] + _split_blocks(t))
      if pool:
        top = _rank_snippets(q, pool, k=10)
        bundle = {"query": q, "snippets": [{"idx": s["i"], "text": s["text"]} for s in top]}
        res = gen_json(GROUNDED_RULES, json.dumps(bundle, ensure_ascii=False), model=MODEL) or {}
        mer = _repair_mermaid((res or {}).get("mermaid") or "") or _ascii_to_mermaid((res.get("summary") or ""))
        summ = (res or {}).get("summary") or ""
        if not mer:
          seed_text = " ".join(s["text"] for s in bundle["snippets"][:6])
          mer = _skeleton_from_text(seed_text)
        if not _is_plausible_mermaid(mer):
          mer = _default_mermaid()
        srcs = hits
        return JSONResponse({"ok": True, "make_diagram": True, "mermaid": mer, "summary": summ, "sources": srcs})

  if "cia triad" in ql or ("confidentiality" in ql and "integrity" in ql and "availability" in ql):
    mer = _diagram_cia()
    srcs = await _scholar_search("CIA triad confidentiality integrity availability security controls", per=5)
    return JSONResponse({"ok":True,"make_diagram":True,"mermaid":mer,"summary":"CIA triad with concrete controls.","sources":srcs})

  if "zero trust" in ql or "zerotrust" in ql:
    mer = _diagram_zerotrust()
    srcs = await _scholar_search("zero trust architecture policy identity device context continuous verification", per=5)
    return JSONResponse({"ok":True,"make_diagram":True,"mermaid":mer,"summary":"Zero Trust decision path.","sources":srcs})

  dlp_preseed = _diagram_dlp() if ("data loss prevention" in ql or "dlp" in ql) else ""

  try:
    req_labels = []
    m = re.search(r"required\s+labels\s*:\s*(.+)$", q, re.I)
    if m:
      req_labels = [t.strip() for t in re.split(r"[,/;]| and ", m.group(1)) if t.strip()]

    llm_mer, llm_sum = _llm_mermaid_plan(q, req_labels)
    mer = llm_mer if _is_plausible_mermaid(llm_mer) else ""

    if not mer and dlp_preseed:
      mer = dlp_preseed
    if not mer:
      mer = _skeleton_from_text(q)

    make = bool(_is_plausible_mermaid(mer))
    if not make:
      mer = _default_mermaid()
      make = True

    q_norms = _normalize_for_oa(q)
    srcs = await _scholar_search((q_norms[0] if q_norms else q), per=5)
    summary = llm_sum or ("Data Loss Prevention pipeline with policy enforcement and mitigations." if dlp_preseed else "")
    return JSONResponse({"ok":True,"make_diagram":make, "mermaid": mer, "summary": summary, "sources": srcs})

  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)

@router.post("/diagram/ask")
def diagram_ask(payload:dict=Body(...)):
  mer = (payload or {}).get("mermaid") or ""
  q = (payload or {}).get("q") or ""
  if not mer.strip(): return JSONResponse({"ok":False,"error":"No diagram loaded. Plan a diagram first."}, status_code=400)
  if _risky(q) or _risky(mer): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  labels = ", ".join(_labels_from_mermaid(mer))
  try:
    user = "Diagram:\n" + mer + "\n\nYou may reference labels: " + labels + "\n\nQuestion:\n" + q
    a = gen_text(ASK_RULES, user, model=MODEL) or ""
    return JSONResponse({"ok":True,"answer":a.strip()})
  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)

# ---------- AutoChart (LLM → normalized chart spec) ----------
_AUTOCHART_SCHEMA = (
  "Return JSON exactly as: {"
  "  chart_type: 'bar'|'pie'|'line'|'heatmap'|'network'|'geo',"
  "  title: string,"
  "  data: array|object, "
  "  enc: {x?:str,y?:str,color?:str,size?:str,group?:str,time?:str,lat?:str,lon?:str,source?:str,target?:str},"
  "  animation?: {field?:string, duration_ms?:number, mode?:'reveal'|'tween'|'force'|'burst'|'loop'},"
  "  notes?: string,"
  "  warnings?: string[]"
  "}. "
  "Rules: "
  "- Choose the chart_type that best fits the prompt semantics. "
  "- If tabular facts are implied (e.g., categories + values), return a 'data' array of objects with columns used by enc. "
  "- For 'network', return data = {nodes:[{id,label,group?}], edges:[{source,target,weight?}]}. "
  "- For 'geo', return an array with lat/lon and optional value/time fields. Use enc.lat/enc.lon for names. "
  "- Prefer a small dataset (≤ 20 rows, ≤ 15 nodes). "
  "- If the prompt contains time words (year, month, day), include a time field and animation.field. "
  "- No code fences."
)

def _autochart_llm_plan(q: str) -> dict:
  seed_shots = [
    ("Compare 2023 browser share", {
      "chart_type":"pie","title":"Browser Share 2023",
      "enc":{"group":"browser","size":"share"},
      "data":[{"browser":"Chrome","share":63},{"browser":"Safari","share":20},{"browser":"Edge","share":6},{"browser":"Firefox","share":5},{"browser":"Other","share":6}],
      "animation":{"field":"","duration_ms":700}
    }),
    ("Monthly signups by plan with total", {
      "chart_type":"line","title":"Monthly Signups",
      "enc":{"x":"month","y":"signups","color":"plan","time":"month"},
      "data":[
        {"month":"Jan","plan":"Free","signups":120},
        {"month":"Jan","plan":"Pro","signups":40},
        {"month":"Feb","plan":"Free","signups":150},
        {"month":"Feb","plan":"Pro","signups":60}
      ],
      "animation":{"field":"month","duration_ms":800}
    }),
    ("Phishing kill-chain relationships", {
      "chart_type":"network","title":"Phishing Kill-Chain",
      "enc":{"source":"source","target":"target","group":"group"},
      "data":{"nodes":[
                {"id":"Email","label":"Email"},
                {"id":"Link","label":"Malicious Link"},
                {"id":"Page","label":"Fake Login"},
                {"id":"Creds","label":"Stolen Creds"}],
              "edges":[
                {"source":"Email","target":"Link"},
                {"source":"Link","target":"Page"},
                {"source":"Page","target":"Creds"}]},
      "animation":{"duration_ms":900}
    }),
  ]
  prompt = _AUTOCHART_SCHEMA + "\n\nExamples:\n" + "\n".join(
    [f"Q: {a}\nA: {json.dumps(b)}" for a,b in seed_shots]
  )
  spec = gen_json(prompt, q, model=MODEL, temperature=0.2) or {}
  return spec

def _autochart_sanity(spec: dict) -> dict:
  ct = (spec.get("chart_type") or "").lower()
  if ct not in {"bar","pie","line","heatmap","network","geo"}:
    spec["chart_type"] = "bar"
  spec.setdefault("title","AutoChart")
  spec.setdefault("enc",{})
  spec.setdefault("animation",{"duration_ms":800})
  if isinstance(spec.get("data"), list) and len(spec["data"])>200:
    spec["data"] = spec["data"][:200]
  if ct=="network" and isinstance(spec.get("data"), dict):
    nodes = (spec["data"].get("nodes") or [])[:80]
    edges = (spec["data"].get("edges") or [])[:160]
    spec["data"] = {"nodes":nodes,"edges":edges}
  return spec
# ----- Rule-based fallback if LLM fails -----
def _autochart_guess_kind(q: str) -> str:
    t = (q or "").lower()
    if any(k in t for k in ["map", "geo", "lat", "lon", "latitude", "longitude", "city", "country"]):
        return "geo"
    if any(k in t for k in ["network", "graph", "nodes", "edges", "relationships", "kill-chain", "kill chain"]):
        return "network"
    if any(k in t for k in ["confusion matrix", "heatmap", "correlation", "matrix"]):
        return "heatmap"
    if any(k in t for k in ["share", "composition", "parts of", "breakdown", "percent", "percentage", "proportion"]):
        return "pie"
    if any(k in t for k in ["trend", "over time", "monthly", "weekly", "daily", "per year", "timeline"]):
        return "line"
    return "bar"

def _autochart_rule_spec(q: str) -> dict:
    kind = _autochart_guess_kind(q)
    if kind == "geo":
        return {"chart_type":"geo","title":"Locations (rule)","enc":{"lat":"lat","lon":"lon","time":"t","size":"count"},
                "data":[{"city":"NYC","lat":40.7128,"lon":-74.0060,"t":"Mon","count":3},
                        {"city":"NYC","lat":40.7128,"lon":-74.0060,"t":"Tue","count":5},
                        {"city":"LA","lat":34.0522,"lon":-118.2437,"t":"Mon","count":2},
                        {"city":"LA","lat":34.0522,"lon":-118.2437,"t":"Tue","count":6}],
                "animation":{"field":"t","duration_ms":900},
                "notes":"Rule-based fallback (no LLM)."}
    if kind == "network":
        return {"chart_type":"network","title":"Relationship Graph (rule)","enc":{"source":"source","target":"target"},
                "data":{"nodes":[{"id":"A"},{"id":"B"},{"id":"C"},{"id":"D"}],
                        "edges":[{"source":"A","target":"B"},{"source":"A","target":"C"},{"source":"C","target":"D"}]},
                "notes":"Rule-based fallback (no LLM)."}
    if kind == "heatmap":
        return {"chart_type":"heatmap","title":"Heatmap (rule)","enc":{"x":"pred","y":"true","color":"count"},
                "data":[{"true":"Cat","pred":"Cat","count":40},{"true":"Cat","pred":"Dog","count":10},
                        {"true":"Dog","pred":"Cat","count":7},{"true":"Dog","pred":"Dog","count":43}],
                "notes":"Rule-based fallback (no LLM)."}
    if kind == "pie":
        return {"chart_type":"pie","title":"Composition (rule)","enc":{"group":"group","size":"value"},
                "data":[{"group":"A","value":50},{"group":"B","value":30},{"group":"C","value":20}],
                "notes":"Rule-based fallback (no LLM)."}
    if kind == "line":
        return {"chart_type":"line","title":"Trend (rule)","enc":{"x":"month","y":"value","time":"month"},
                "data":[{"month":"Jan","value":10},{"month":"Feb","value":14},{"month":"Mar","value":12},{"month":"Apr","value":18}],
                "animation":{"field":"month","duration_ms":800},
                "notes":"Rule-based fallback (no LLM)."}
    return {"chart_type":"bar","title":"Bars (rule)","enc":{"x":"label","y":"value"},
            "data":[{"label":"X","value":7},{"label":"Y","value":5},{"label":"Z","value":9}],
            "notes":"Rule-based fallback (no LLM)."}


@router.post("/autochart/plan")
def autochart_plan(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  if _risky(q):
    return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  try:
    spec = _autochart_llm_plan(q)
    spec = _autochart_sanity(spec)
    return JSONResponse({"ok":True,"spec":spec})
  except Exception as e:
    fb = {
      "chart_type":"bar","title":"Fallback Bar",
      "enc":{"x":"category","y":"value"},
      "data":[{"category":"A","value":5},{"category":"B","value":3},{"category":"C","value":7}],
      "animation":{"duration_ms":700},
      "notes":"LLM error; fallback synthetic chart."
    }
    return JSONResponse({"ok":True,"spec":fb,"fallback":"llm_error","error":str(e)})


@router.post("/autochart/sample")
def autochart_sample(payload:dict=Body(...)):
  kind = ((payload or {}).get("kind") or "bar").lower()
  if kind == "geo":
      spec = {
          "chart_type": "geo",
          "title": "Incidents by City (Animated by Day)",
          "enc": {"lat": "lat", "lon": "lon", "time": "day", "size": "count"},
          "data": [
              # Monday
              {"city": "NYC", "lat": 40.7128, "lon": -74.0060, "day": "Mon", "count": 4},
              {"city": "LA", "lat": 34.0522, "lon": -118.2437, "day": "Mon", "count": 2},
              {"city": "SF", "lat": 37.7749, "lon": -122.4194, "day": "Mon", "count": 1},
              # Tuesday
              {"city": "NYC", "lat": 40.7128, "lon": -74.0060, "day": "Tue", "count": 6},
              {"city": "LA", "lat": 34.0522, "lon": -118.2437, "day": "Tue", "count": 5},
              {"city": "SF", "lat": 37.7749, "lon": -122.4194, "day": "Tue", "count": 2},
              # Wednesday
              {"city": "NYC", "lat": 40.7128, "lon": -74.0060, "day": "Wed", "count": 7},
              {"city": "LA", "lat": 34.0522, "lon": -118.2437, "day": "Wed", "count": 4},
              {"city": "SF", "lat": 37.7749, "lon": -122.4194, "day": "Wed", "count": 3}
          ],
          "animation": {"field": "day", "duration_ms": 900, "mode": "reveal"}
      }
  elif kind == "network":
    spec={"chart_type":"network","title":"Mini Net","enc":{"source":"source","target":"target"},
         "data":{"nodes":[{"id":"A"},{"id":"B"},{"id":"C"}],
                 "edges":[{"source":"A","target":"B"},{"source":"B","target":"C"}]}}
  elif kind == "heatmap":
    spec={"chart_type":"heatmap","title":"Confusion Matrix","enc":{"x":"pred","y":"true","color":"count"},
         "data":[{"true":"Cat","pred":"Cat","count":42},{"true":"Cat","pred":"Dog","count":8},
                 {"true":"Dog","pred":"Cat","count":5},{"true":"Dog","pred":"Dog","count":45}]}
  else:
      spec = {"chart_type": "bar", "title": "Bars",
              "enc": {"x": "label", "y": "value", "time": "month"},
              "data": [{"label": "X", "value": 5, "month": "Jan"},
                       {"label": "Y", "value": 8, "month": "Feb"},
                       {"label": "Z", "value": 3, "month": "Mar"}],
              "animation": {"field": "month", "duration_ms": 900, "mode": "reveal"}}

  return JSONResponse({"ok":True,"spec":_autochart_sanity(spec)})

# Put near other helpers in studio_routes.py


# ---------- notebook & docs ----------
@router.get("/notebook/templates")
def notebook_templates():
  tpls = [
    {"id":"cycle_check","label":"Graph sanity (cycles/isolates)","code":
     "import json, networkx as nx\nARTIFACTS['checks']={}\nG=nx.DiGraph()\n"
     "for n in spec.get('nodes',[]): G.add_node(n.get('id') or n.get('label'))\n"
     "for e in spec.get('edges',[]):\n"
     "    a,b=(e[0],e[1]) if isinstance(e,(list,tuple)) else (e.get('source'),e.get('target'))\n"
     "    if a and b: G.add_edge(a,b)\n"
     "ARTIFACTS['checks']['node_count']=G.number_of_nodes()\n"
     "ARTIFACTS['checks']['edge_count']=G.number_of_edges()\n"
     "ARTIFACTS['checks']['is_dag']=nx.is_directed_acyclic_graph(G)\n"
     "ARTIFACTS['checks']['isolated']=[n for n in G.nodes() if G.degree(n)==0]\n"
     "ARTIFACTS['verdicts']=(['OK'] if ARTIFACTS['checks']['is_dag'] and not ARTIFACTS['checks']['isolated'] else [])\n"
     "if not ARTIFACTS['checks']['is_dag']: ARTIFACTS.setdefault('verdicts',[]).append('Has cycle(s)')\n"
     "if ARTIFACTS['checks']['isolated']: ARTIFACTS.setdefault('verdicts',[]).append('Isolated nodes present')\n"},
    {"id":"sources_to_table","label":"Show sources as a table","code":
     "import pandas as pd\nARTIFACTS['sources_table']=pd.DataFrame(SOURCES).to_dict(orient='records')"},
    {"id":"edge_trace","label":"Trace edges to snippets","code":
     "import json\n"
     "edges = []\n"
     "for e in spec.get('edges',[]):\n"
     "  if isinstance(e,(list,tuple)):\n"
     "    a,b = e[0], e[1]\n"
     "    lbl = e[2] if len(e)>2 else ''\n"
     "  else:\n"
     "    a,b,lbl = e.get('source'),e.get('target'),e.get('label','')\n"
     "  edges.append({'from':a,'to':b,'label':lbl})\n"
     "ARTIFACTS['edges']=edges\n"
     "ARTIFACTS['note']='Manually map each edge to citations [#] from the UI summary.'"}
  ]
  return JSONResponse({"ok":True,"templates":tpls})

@router.post("/notebook/run")
def notebook_run(payload:dict=Body(...)):
  code = (payload or {}).get("code") or ""
  spec = (payload or {}).get("spec") or {}
  sources = (payload or {}).get("sources") or []
  if _risky(json.dumps(spec)): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  nb = NotebookRunner(timeout_s=10)
  try:
    pre  = "ARTIFACTS={}\n"
    pre += "spec=" + json.dumps(spec) + "\n"
    pre += "SOURCES=" + json.dumps(sources) + "\n"
    res = nb.run([pre + code])
    return JSONResponse({"ok":True,"artifacts":res.artifacts or {}})
  except Exception as e:
    return JSONResponse({"ok":False,"error":str(e)}, status_code=400)
  finally:
    nb.stop()

@router.post("/docs/upload")
async def docs_upload(file: UploadFile = File(...)):
  try:
    b = await file.read()
    text = ""
    if file.filename.lower().endswith(".pdf"):
      reader = PdfReader(io.BytesIO(b))
      for p in reader.pages[:12]:
        text += (p.extract_text() or "") + "\n"
    else:
      try: text = b.decode("utf-8", errors="ignore")[:10000]
      except Exception: text = ""
    if not text.strip(): text="(no extractable text)"
    _INGEST["text"]=text; _INGEST["images"]=[]
    return JSONResponse({"ok":True,"chars":len(text)})
  except Exception as e:
    return JSONResponse({"ok":False,"error":str(e)}, status_code=400)

@router.post("/docs/upload_image")
async def docs_upload_image(file: UploadFile = File(...)):
  try:
    b = await file.read()
    b64 = "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    _INGEST["images"].append(b64)
    return JSONResponse({"ok":True,"images":len(_INGEST['images'])})
  except Exception as e:
    return JSONResponse({"ok":False,"error":str(e)}, status_code=400)

@router.post("/docs/summarize")
async def docs_summarize(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  if _risky(q): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  body = "Document excerpt:\n" + (_INGEST["text"][:4000] or "(empty)") + f"\nImages: {len(_INGEST['images'])}\n"
  q_norms = _normalize_for_oa(q)
  srcs = await _scholar_search((q_norms[0] if q_norms else (q or "cyber security")), per=5)
  try:
    out = gen_text(DOC_RULES + "\nCite sources inline by [#] where useful.", body, model=MODEL) or ""
    return JSONResponse({"ok":True,"summary":out.strip(),"sources":srcs})
  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)

@router.post("/export")
def export_image(payload: dict = Body(...)):
  svg_text = (payload or {}).get("svg") or ""
  fmt = ((payload or {}).get("format") or "svg").lower()
  scale = float((payload or {}).get("scale") or 1.0)
  brand = (payload or {}).get("brand") or None
  fname = (payload or {}).get("filename") or "diagram"

  if not svg_text.strip():
    return JSONResponse({"ok": False, "error": "No SVG provided"}, status_code=400)

  svg_text = _inject_brand_css(svg_text, brand)

  if fmt == "svg":
    return Response(
      content=svg_text.encode("utf-8"),
      media_type="image/svg+xml",
      headers={"Content-Disposition": f'attachment; filename="{fname}.svg"'}
    )

  elif fmt == "png":
    try:
      import cairosvg
      png_bytes = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), scale=scale)
      return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{fname}.png"'}
      )
    except Exception as e:
      return JSONResponse({"ok": False, "error": f"PNG convert: {e}"}, status_code=500)

  else:
    return JSONResponse({"ok": False, "error": f"Unknown format: {fmt}"}, status_code=400)

@router.get("/scholarviz", response_class=HTMLResponse)
def scholarviz(request: Request, q: str = Query("", description="Ask the model")):
  page = """
<!doctype html><html><head>
<meta charset="utf-8"/><title>ScholarViz</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1/dist/leaflet.css"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://unpkg.com/leaflet@1/dist/leaflet.js"></script>
<style>
body{font-family:ui-sans-serif,system-ui,-apple-system;margin:0} header{background:#0f172a;color:#fff;padding:12px 16px}
main{max-width:960px;margin:0 auto;padding:16px} .card{border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:12px 0}
textarea,input,button,select{font-size:14px} textarea,input{border:1px solid #e5e7eb;border-radius:8px;padding:8px}
button{background:#111827;color:#fff;border:none;border-radius:8px;padding:8px 12px;cursor:pointer} button:hover{background:#0f172a}
#diagram_view svg{max-width:100%} .src a{text-decoration:none}
</style>
<script>mermaid.initialize({startOnLoad:false});</script>
</head><body>
<header><h1 style="margin:0;font-size:18px">ScholarViz</h1></header>
<main>

<div class="card">
  <h3 style="margin:0 0 8px">Ask the LLM</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <input id="ask_q" placeholder="e.g., Summarize Zero Trust in 120 words" style="flex:1;min-width:240px"/>
    <button id="ask_go">Ask</button>
  </div>
  <pre id="ask_out" style="white-space:pre-wrap;margin-top:10px"></pre>
  <div id="ask_src" class="src" style="margin-top:6px"></div>
  <button id="ask_to_nb" style="margin-top:6px;display:none">Send sources to Notebook</button>
</div>

<div class="card">
  <h3 style="margin:0 0 8px">Plan a Diagram</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <input id="plan_q" placeholder="e.g., CIA triad with explicit labels" style="flex:1;min-width:240px"/>
    <button id="plan_go">Plan</button>
    <button id="plan_grounded">Grounded Plan (from PDFs)</button>
  </div>
  <p id="plan_sum" style="color:#6b7280"></p>
  <div id="plan_src" class="src" style="margin-top:6px"></div>

  <pre id="plan_g_out" style="white-space:pre-wrap;margin-top:10px"></pre>

  <div id="diagram_view" class="card" style="display:none"></div>
  <details id="mer_raw_wrap" style="display:none;margin-top:8px">
    <summary>Show raw Mermaid</summary>
    <pre id="mer_raw" style="white-space:pre-wrap"></pre>
  </details>

  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px">
    <select id="brand_sel">
      <option value="">No brand</option>
      <option value='{"primary":"#0ea5e9","stroke":"#0f172a","font":"Inter, ui-sans-serif","rounded":true}'>Sky/Slate</option>
      <option value='{"primary":"#22c55e","stroke":"#052e16","font":"Inter, ui-sans-serif","rounded":true}'>Green Lab</option>
      <option value='{"primary":"#a78bfa","stroke":"#312e81","font":"Inter, ui-sans-serif","rounded":true}'>Iris</option>
    </select>
    <button id="exp_svg" disabled>Export SVG</button>
    <button id="exp_png" disabled>Export PNG</button>
    <input id="exp_name" placeholder="diagram filename" style="min-width:160px"/>
    <input id="exp_scale" type="number" step="0.1" value="1.0" title="PNG scale" style="width:90px"/>
  </div>

  <div id="fig_thumbs" style="margin-top:10px"></div>
</div>

<div class="card">
  <h3 style="margin:0 0 8px">AutoChart (model picks chart + animates)</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <input id="ac_q" placeholder="e.g., Show 2024 monthly malware detections by family" style="flex:1;min-width:280px"/>
  <button id="ac_plan">Plan</button>
  <select id="ac_kind">
    <option value="bar">bar</option>
    <option value="line">line</option>
    <option value="pie">pie</option>
    <option value="heatmap">heatmap</option>
    <option value="network">network</option>
    <option value="geo" selected>geo</option>
  </select>
  <button id="ac_sample">Sample</button>
</div>
  <pre id="ac_meta" style="white-space:pre-wrap;margin-top:8px;color:#6b7280"></pre>
  <div id="ac_plot" style="width:100%;min-height:360px;border:1px solid #e5e7eb;border-radius:12px;margin-top:8px;display:none"></div>
  <div id="ac_map"  style="width:100%;height:420px;border:1px solid #e5e7eb;border-radius:12px;margin-top:8px;display:none"></div>
  <div id="ac_net"  style="width:100%;height:420px;border:1px solid #e5e7eb;border-radius:12px;margin-top:8px;display:none"></div>
</div>

<div class="card">
  <h3 style="margin:0 0 8px">Animated Lesson</h3>

  <div id="diagram_cy" style="height:360px;border:1px solid #e5e7eb;border-radius:12px;display:none;margin-bottom:10px"></div>

  <div id="lesson" style="margin-top:6px">
    <div id="lessonTitle" style="font-weight:600;margin-bottom:8px;"></div>
    <div id="lessonNote" style="margin:8px 0 12px;"></div>
    <div id="lessonQuiz" style="display:none;margin-top:8px;"></div>
    <div style="margin-top:10px;">
      <button id="lessonPrev">◀</button>
      <button id="lessonNext">▶</button>
      <button id="lessonStart" style="margin-left:8px">Animate Lesson</button>
      <small style="color:#6b7280;margin-left:8px">Uses the text in “Plan a Diagram”</small>
    </div>
  </div>
</div>

<div class="card">
  <h3 style="margin:0 0 8px">Ask about this Diagram</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <input id="d_q" placeholder="e.g., Where does WebAuthn help most?" style="flex:1;min-width:240px"/>
    <button id="d_go" disabled>Ask</button>
  </div>
  <pre id="d_out" style="white-space:pre-wrap;margin-top:10px"></pre>
</div>

<div class="card">
  <h3 style="margin:0 0 8px">Notebook + Papers/Diagrams</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <input id="doc_file" type="file"/><button id="doc_up">Upload PDF/Text</button>
    <input id="img_file" type="file" accept="image/*"/><button id="img_up">Upload Diagram</button>
    <input id="doc_q" placeholder="Ask about the uploaded material…" style="flex:1;min-width:240px"/>
    <button id="doc_sum">Summarize/Answer</button>
  </div>
  <pre id="doc_out" style="white-space:pre-wrap;margin-top:10px"></pre>
  <div id="doc_src" class="src" style="margin-top:6px"></div>

  <div class="card" style="margin-top:10px">
    <h4 style="margin:0 0 8px">Notebook</h4>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <select id="nb_tpl"><option value="">Templates…</option></select><button id="nb_load">Load</button>
    </div>
    <textarea id="nb_code" style="width:100%;height:160px;margin-top:8px"></textarea>
    <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
      <button id="nb_run">Run</button>
      <input id="nb_spec" placeholder="(optional) spec JSON" style="flex:1;min-width:240px"/>
    </div>
    <pre id="nb_out" style="white-space:pre-wrap;margin-top:10px"></pre>
  </div>
</div>

</main>
<script>
let CURRENT_MERMAID = null;
let LAST_SOURCES = [];

async function post(path, body, form){
  const res = await fetch(path, {method:'POST', headers: form?undefined:{'Content-Type':'application/json'}, body: form?JSON.stringify(body||{}):JSON.stringify(body||{})});
  return res.json();
}
function renderSources(el, src){
  if (!src || !src.length){ el.innerHTML=''; return; }
  el.innerHTML = '<strong>Sources:</strong><ol>' + src.map((s,i)=>(`<li><a href="${s.url||'#'}" target="_blank">[${i+1}] ${s.title} (${s.year||''})</a>` + (s.authors?` — <span style="color:#6b7280">${s.authors}</span>`:'') + (s.venue?` <em style="color:#6b7280">· ${s.venue}</em>`:'') + `</li>`)).join('') + '</ol>';
}

function currentSVGString(){
  const el = document.querySelector('#diagram_view svg');
  if (!el) return '';
  let s = el.outerHTML;
  if (!/xmlns=/.test(s)) s = s.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
  return s;
}

function teardownLesson(){
  const cyDiv = document.getElementById('diagram_cy');
  if (cyDiv){
    cyDiv.style.display = 'none';
    const dim = document.getElementById('lessonDim');
    if (dim) dim.remove();
    if (window.cy && typeof window.cy.destroy === 'function'){
      try { window.cy.destroy(); } catch(_){}
    }
    cyDiv.innerHTML = '';
  }
  const merDiv = document.getElementById('diagram_view');
  if (merDiv) merDiv.style.display = 'block';
}

async function exportDiagram(fmt){
  const svg = currentSVGString();
  if (!svg){ alert('No rendered diagram.'); return; }
  const brandSel = document.getElementById('brand_sel');
  let brand = null;
  try { brand = brandSel && brandSel.value ? JSON.parse(brandSel.value) : null; } catch(_){}
  const filename = (document.getElementById('exp_name').value || 'diagram').trim();
  const scale = parseFloat(document.getElementById('exp_scale').value || '1.0');
  const res = await fetch('/export', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ svg, format: fmt, scale, brand, filename })
  });
  if (!res.ok){
    const j = await res.json().catch(()=> ({}));
    alert('Export failed: ' + (j.error || res.statusText));
    return;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename + '.' + (fmt==='png'?'png':'svg');
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

document.getElementById('exp_svg').onclick = ()=> exportDiagram('svg');
document.getElementById('exp_png').onclick = ()=> exportDiagram('png');

document.getElementById('ask_go').onclick = async ()=>{
  const q = document.getElementById('ask_q').value.trim();
  document.getElementById('ask_out').textContent='…';
  const d = await post('/ask', {q});
  if (!d.ok){ document.getElementById('ask_out').textContent='Error: '+d.error; return; }
  document.getElementById('ask_out').textContent = (d.answer||'(no answer)');
  LAST_SOURCES = d.sources||[];
  renderSources(document.getElementById('ask_src'), LAST_SOURCES);
  document.getElementById('ask_to_nb').style.display = LAST_SOURCES.length? 'inline-block':'none';
};
document.getElementById('ask_to_nb').onclick = ()=>{
  const code = "import pandas as pd\\nARTIFACTS['sources_table']=pd.DataFrame(SOURCES).to_dict(orient='records')";
  document.getElementById('nb_code').value = code;
  document.getElementById('nb_out').textContent = "Loaded a cell to show sources as a table. Click Run.";
};

document.getElementById('plan_go').onclick = async ()=>{
  teardownLesson();
  const q = document.getElementById('plan_q').value.trim();
  document.getElementById('plan_sum').textContent='…';
  const d = await post('/diagram/plan', {q});
  if (!d.ok){ document.getElementById('plan_sum').textContent='Error: '+d.error; return; }
  document.getElementById('plan_sum').textContent = d.summary||'';
  renderSources(document.getElementById('plan_src'), d.sources||[]);
  const view = document.getElementById('diagram_view');
  const rawWrap = document.getElementById('mer_raw_wrap');
  const rawPre  = document.getElementById('mer_raw');
  if (d.make_diagram && d.mermaid){
    CURRENT_MERMAID = d.mermaid;
    view.style.display='block'; view.textContent='';
    rawWrap.style.display = 'none'; rawPre.textContent = '';
    mermaid.render('schviz_svg', d.mermaid).then(r=>{
      view.innerHTML=r.svg;
      document.getElementById('exp_svg').disabled = false;
      document.getElementById('exp_png').disabled = false;
    }).catch(_=>{
      view.textContent='Mermaid error';
      rawPre.textContent = d.mermaid || '(empty)';
      rawWrap.style.display = 'block';
    });
    document.getElementById('d_go').disabled = false;
  } else {
    CURRENT_MERMAID = null; view.style.display='none';
    rawWrap.style.display='none'; rawPre.textContent='';
    document.getElementById('d_go').disabled = true;
  }
};

document.getElementById('plan_grounded').onclick = async ()=>{
  teardownLesson();
  const q = document.getElementById('plan_q').value.trim();
  const out = document.getElementById('plan_g_out');
  const view = document.getElementById('diagram_view');
  const rawWrap = document.getElementById('mer_raw_wrap');
  const rawPre  = document.getElementById('mer_raw');
  const figsDiv = document.getElementById('fig_thumbs');
  out.textContent = 'Fetching OA PDFs and synthesizing…';
  figsDiv.innerHTML = '';
  try{
    const d = await fetch('/diagram/grounded', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ q, n: 3, diversify: true, seed: Date.now() % 1e9, temp: 0.6 })
    }).then(r=>r.json());

    if (!d.ok){
      out.textContent = 'Grounded synthesis failed.';
      CURRENT_MERMAID = null;
      document.getElementById('d_go').disabled = true;
      return;
    }

    out.textContent = (d.summary||'');
    if (d.fallback) out.textContent = '(Fallback: '+d.fallback+')\\n' + out.textContent;

    if (d.mermaid){
      CURRENT_MERMAID = d.mermaid;
      view.style.display='block'; view.textContent='';
      rawWrap.style.display = 'none'; rawPre.textContent = '';
      mermaid.render('schviz_gsvg', d.mermaid)
        .then(r=>{
          view.innerHTML=r.svg;
          document.getElementById('exp_svg').disabled = false;
          document.getElementById('exp_png').disabled = false;
        })
        .catch(_=>{
          view.textContent='Mermaid error';
          rawPre.textContent = d.mermaid || '(empty)';
          rawWrap.style.display = 'block';
        });
      document.getElementById('d_go').disabled=false;

      (function season(){
        const svg = document.querySelector('#diagram_view svg');
        if (!svg) return;
        const seed = Math.abs([...Date.now().toString()].reduce((a,c)=>a + c.charCodeAt(0),0)) % 9973;
        const jitter = 0.25 + (seed % 20)/100;
        svg.querySelectorAll('.edgePath path').forEach(p => p.setAttribute('stroke-width', (1.2*jitter).toFixed(2)));
        svg.querySelectorAll('.node rect').forEach(r => {
          const base = 10 + (seed % 6);
          r.setAttribute('rx', base); r.setAttribute('ry', base);
        });
      })();
    } else {
      CURRENT_MERMAID = null;
      view.style.display='none';
      rawWrap.style.display='none';
      rawPre.textContent='';
      document.getElementById('d_go').disabled=true;
    }

    renderSources(document.getElementById('plan_src'), d.sources||[]);

    const lis = (d.snippets||[])
      .map((s,i)=>`[${i}] ${s.text.substring(0,300)}${s.text.length>300?'…':''}`)
      .join('\\n\\n');
    out.textContent += lis ? `\\n\\nTop snippets:\\n${lis}` : '';

    if (d.figures && d.figures.length){
      const imgs = d.figures.map(f=>`<div style="display:inline-block;margin:6px">
        <div style="font-size:12px;color:#6b7280">p.${f.page} — ${(f.paper||'')}</div>
        <img src="${f.png_b64}" style="max-width:160px;display:block;border:1px solid #e5e7eb;border-radius:6px"/>
        <div style="max-width:160px;font-size:12px;color:#374151">${(f.caption||'').slice(0,160)}</div>
      </div>`).join('');
      figsDiv.innerHTML = "<div style='margin-top:8px'><strong>Figures used:</strong><br/>"+imgs+"</div>";
    }

  } catch(e){
    out.textContent = 'Error: ' + e.message;
  }
};

document.getElementById('d_go').onclick = async ()=>{
  const q = document.getElementById('d_q').value.trim();
  const out = document.getElementById('d_out');
  if (!CURRENT_MERMAID){ out.textContent = "No diagram loaded. Plan a diagram first."; return; }
  out.textContent='…';
  const d = await post('/diagram/ask', { q, mermaid: CURRENT_MERMAID });
  out.textContent = d.ok ? (d.answer||'(no answer)') : ('Error: '+d.error);
};

document.getElementById('doc_up').onclick = async ()=>{
  const f = document.getElementById('doc_file').files[0];
  if (!f){ document.getElementById('doc_out').textContent='Choose a PDF/text file'; return; }
  const fd = new FormData(); fd.append('file', f);
  document.getElementById('doc_out').textContent='Uploading…';
  const d = await post('/docs/upload', fd, true);
  document.getElementById('doc_out').textContent = d.ok? ('Uploaded. Extracted chars: '+d.chars):('Error: '+d.error);
};

document.getElementById('img_up').onclick = async ()=>{
  const f = document.getElementById('img_file').files[0];
  if (!f){ document.getElementById('doc_out').textContent='Choose an image'; return; }
  const fd = new FormData(); fd.append('file', f);
  document.getElementById('doc_out').textContent='Uploading image…';
  const d = await post('/docs/upload_image', fd, true);
  document.getElementById('doc_out').textContent = d.ok? ('Images stored: '+d.images):('Error: '+d.error);
};

document.getElementById('doc_sum').onclick = async ()=>{
  const q = document.getElementById('doc_q').value.trim();
  document.getElementById('doc_out').textContent='Summarizing…';
  const d = await fetch('/docs/summarize', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({q})}).then(r=>r.json());
  document.getElementById('doc_out').textContent = d.ok? (d.summary||'(no summary)'):('Error: '+d.error);
  renderSources(document.getElementById('doc_src'), d.sources||[]);
};

(async function nbInit(){
  const sel=document.getElementById('nb_tpl'), load=document.getElementById('nb_load'), run=document.getElementById('nb_run'), out=document.getElementById('nb_out');
  if (!sel||!load||!run) return;
  try{
    const d=await fetch('/notebook/templates').then(r=>r.json());
    if (d.ok){ d.templates.forEach(t=>{ const o=document.createElement('option'); o.value=t.id; o.textContent=t.label; o.dataset.code=t.code; sel.appendChild(o); }); }
  }catch(_){}
  load.onclick=()=>{ const o=sel.options[sel.selectedIndex]; const c=o&&o.dataset.code; if(c) document.getElementById('nb_code').value=c; };
  run.onclick=async ()=>{
    const code=document.getElementById('nb_code').value.trim();
    const specRaw=document.getElementById('nb_spec').value.trim();
    let spec={}; if (specRaw){ try{ spec=JSON.parse(specRaw);}catch(_){ }}
    out.textContent='Running…';
    const d=await post('/notebook/run', {code, spec, sources: LAST_SOURCES });
    out.textContent = d.ok? JSON.stringify(d.artifacts||{}, null, 2) : ('Error: '+d.error);
  };
})();

// ---------- AutoChart client ----------
let AC_SPEC = null;
let AC_MAP = null;
let AC_NET = null;

function acHideAll(){
  document.getElementById('ac_plot').style.display='none';
  document.getElementById('ac_map').style.display='none';
  document.getElementById('ac_net').style.display='none';
}

function acMeta(spec){
  const w = (spec.warnings||[]).join('; ');
  return (spec.title? ('Title: '+spec.title+'\\n'):'') +
         ('Type: '+spec.chart_type+'\\n') +
         (spec.notes? ('Notes: '+spec.notes+'\\n'):'') +
         (w? ('Warnings: '+w+'\\n'):'');
}

function acUnique(vals){ return [...new Set(vals)]; }
function acGroupBy(arr, key){ const m={}; arr.forEach(r=>{const k=r[key]; (m[k]||(m[k]=[])).push(r);}); return m; }

function acToPlotly(spec){
  const enc = spec.enc||{};
  const data = spec.data||[];
  const tfield = (spec.animation && spec.animation.field) || enc.time || "";
  const duration = (spec.animation && spec.animation.duration_ms) || 800;
  const mode = (spec.animation && spec.animation.mode) || "reveal";

  // No time? simple static plots
  if (!tfield){
    if (spec.chart_type==='bar'){
      const x= data.map(r=>r[enc.x]); const y= data.map(r=>r[enc.y]);
      return {traces:[{type:'bar', x, y}], layout:{title:spec.title}, frames:[], duration};
    }
    if (spec.chart_type==='line'){
      const groups = enc.color? acUnique(data.map(r=>r[enc.color])): [null];
      const traces = groups.map(g=>{
        const rows = data.filter(r=> (g===null? true : r[enc.color]===g));
        return {type:'scatter', mode:'lines+markers', name:g||undefined,
                x: rows.map(r=>r[enc.x]), y: rows.map(r=>r[enc.y])};
      });
      return {traces, layout:{title:spec.title}, frames:[], duration};
    }
    if (spec.chart_type==='pie'){
      const labels = data.map(r=> r[enc.group] || r[enc.x]);
      const values = data.map(r=> r[enc.size]  || r[enc.y]);
      return {traces:[{type:'pie', labels, values, textinfo:'label+percent'}],
              layout:{title:spec.title}, frames:[], duration};
    }
    if (spec.chart_type==='heatmap'){
      const xs=acUnique(data.map(r=>r[enc.x])), ys=acUnique(data.map(r=>r[enc.y]));
      const Z = ys.map(_=> xs.map(__=>0));
      data.forEach(r=>{const xi=xs.indexOf(r[enc.x]); const yi=ys.indexOf(r[enc.y]); Z[yi][xi] = r[enc.color] ?? r.value ?? 0;});
      return {traces:[{type:'heatmap', x:xs, y:ys, z:Z}],
              layout:{title:spec.title}, frames:[], duration};
    }
    return {traces:[], layout:{title:spec.title}, frames:[], duration};
  }

  // Time-aware: build frames
  const times = acUnique(data.map(r=> r[tfield]));
  const byTime = acGroupBy(data, tfield);
  let traces=[], frames=[];

  if (spec.chart_type==='bar'){
    const cats = acUnique(data.map(r=> r[enc.x]));
    traces = [{type:'bar', x: cats, y: cats.map(_=>0)}];
    frames = times.map(t=>{
      const rows = byTime[t]||[];
      const y = cats.map(c=> (rows.find(r=>r[enc.x]===c)||{})[enc.y] || 0);
      return {name:String(t), data:[{x:cats, y}]};
    });
  }
  else if (spec.chart_type==='line'){
    const groups = enc.color? acUnique(data.map(r=>r[enc.color])): [null];
    traces = groups.map(g=> ({type:'scatter', mode:'lines+markers', name:g||undefined, x:[], y:[]}));
    frames = times.map(t=>{
      const rows = byTime[t]||[];
      const payload = groups.map(g=>{
        const slice = rows.filter(r=> (g===null? true : r[enc.color]===g));
        return {x: slice.map(r=> r[enc.x]), y: slice.map(r=> r[enc.y])};
      });
      return {name:String(t), data: payload};
    });
  }
  else if (spec.chart_type==='pie'){
    const cats = acUnique(data.map(r=> r[enc.group] || r[enc.x]));
    traces = [{type:'pie', labels: cats, values: cats.map(_=>0), textinfo:'label+percent'}];
    frames = times.map(t=>{
      const rows = byTime[t]||[];
      const vals = cats.map(c=>{
        const row = rows.find(r=> (r[enc.group]||r[enc.x])===c);
        return (row && (row[enc.size]||row[enc.y])) || 0;
      });
      return {name:String(t), data:[{labels:cats, values:vals}]};
    });
  }
  else if (spec.chart_type==='heatmap'){
    const xs=acUnique(data.map(r=>r[enc.x])), ys=acUnique(data.map(r=>r[enc.y]));
    traces = [{type:'heatmap', x:xs, y:ys, z: ys.map(_=> xs.map(__=>0))}];
    frames = times.map(t=>{
      const rows = byTime[t]||[];
      const Z = ys.map(_=> xs.map(__=>0));
      rows.forEach(r=>{
        const xi=xs.indexOf(r[enc.x]); const yi=ys.indexOf(r[enc.y]);
        Z[yi][xi] = r[enc.color] ?? r.value ?? 0;
      });
      return {name:String(t), data:[{x:xs, y:ys, z:Z}]};
    });
  }

  const layout = {
    title: spec.title,
    updatemenus: [{
      type:'buttons', x:0.02, y:1.15, xanchor:'left',
      buttons: [
        {label:'Play', method:'animate',
         args: [null, {fromcurrent:true, transition:{duration:duration*0.4},
                       frame:{duration:duration, redraw:true}, mode:'immediate'}]},
        {label:'Pause', method:'animate',
         args: [[null], {mode:'immediate', frame:{duration:0, redraw:false}, transition:{duration:0}}]}
      ]
    }],
    sliders: [{
      active:0, x:0.12, len:0.8, y:1.12,
      currentvalue:{prefix: (tfield||'time')+': '},
      steps: times.map(t=>({label:String(t), method:'animate',
        args:[[String(t)], {mode:'immediate', frame:{duration:duration, redraw:true},
                            transition:{duration:duration*0.4}}]}))
    }]
  };

  return {traces, layout, frames, duration};
}

function acAnimatePlotly(elId, pack){
  const el = document.getElementById(elId);
  el.style.display='block';
  Plotly.newPlot(el, pack.traces, pack.layout, {displayModeBar:true, responsive:true}).then(()=>{
    if (pack.frames && pack.frames.length){
      Plotly.addFrames(el, pack.frames);
    }
  });
}


function acRenderNetwork(spec){
  acHideAll();
  const div = document.getElementById('ac_net');
  div.style.display='block';
  if (window.AC_NET && typeof window.AC_NET.destroy==='function'){ try{window.AC_NET.destroy();}catch(_){} }

  const nodes = (spec.data.nodes||[]).map(n=> ({ data:{ id:n.id, label:n.label||n.id, group:n.group||'' }}));
  const edges = (spec.data.edges||[]).map(e=> ({ data:{ source:e.source, target:e.target, weight:e.weight||1 }}));

  window.AC_NET = cytoscape({
    container: div,
    elements: [...nodes, ...edges],
    style: [
      { selector: 'node',
        style: {
          'label':'data(label)','text-valign':'center',
          'background-color':'#60a5fa','border-width':1,'border-color':'#0f172a',
          'width':28,'height':28,'font-size':10
        }},
      { selector: 'edge',
        style: {'width':2,'line-color':'#94a3b8','curve-style':'bezier',
                'target-arrow-shape':'triangle','target-arrow-color':'#94a3b8'} },
      { selector: 'node.pulse', style: { 'width':36, 'height':36 } }
    ],
    layout: { name:'cose', animate:true, animationDuration:500 }
  });

  // gentle continuous motion + pulsing
  let tick = 0;
  const jiggle = setInterval(()=>{
    if (!window.AC_NET) return clearInterval(jiggle);
    tick++;
    window.AC_NET.layout({name:'cose', animate:true, animationDuration:500}).run();
    window.AC_NET.nodes().removeClass('pulse');
    window.AC_NET.nodes().filter((n,i)=> (i + tick) % 7 === 0).addClass('pulse');
  }, 1800);

  window.AC_NET.on('mouseover','node', e=>{
    const n = e.target;
    n.connectedEdges().animate({style:{'line-color':'#1f2937','target-arrow-color':'#1f2937'}},{duration:200});
  });
  window.AC_NET.on('mouseout','node', e=>{
    const n = e.target;
    n.connectedEdges().animate({style:{'line-color':'#94a3b8','target-arrow-color':'#94a3b8'}},{duration:200});
  });
}


function acRenderGeo(spec){
  acHideAll();
  const div = document.getElementById('ac_map');
  div.style.display='block';
  const data = spec.data||[];
  const enc = spec.enc||{};
  const latK = enc.lat||'lat', lonK=enc.lon||'lon';
  const timeField = (spec.animation && spec.animation.field) || enc.time || null;
  const dur = (spec.animation && spec.animation.duration_ms) || 900;

  if (!window.AC_MAP){
    window.AC_MAP = L.map('ac_map').setView([20,0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19}).addTo(window.AC_MAP);
  }
  window.AC_MAP.eachLayer(l=>{ if (l instanceof L.CircleMarker) window.AC_MAP.removeLayer(l); });

  function drawPoints(rows){
    const key = (r)=> `${+r[latK].toFixed(4)},${+r[lonK].toFixed(4)}`;
    const buckets = {};
    rows.forEach(r=>{
      const la=r[latK], lo=r[lonK];
      if (la==null || lo==null) return;
      const k = key(r);
      (buckets[k]||(buckets[k]=[])).push(r);
    });
    Object.entries(buckets).forEach(([k, arr])=>{
      const [la, lo] = k.split(',').map(Number);
      const total = arr.reduce((s,r)=> s + (+r[enc.size] || +r[enc.color] || +r.value || 1), 0);
      const rad = Math.max(5, Math.min(16, 4 + Math.sqrt(total)));
      const label = (arr[0].city||'point') + (arr.length>1? ` (+${arr.length-1})`:'') + (total?`: ${total}`:'');
      L.circleMarker([la, lo], {radius: rad}).addTo(window.AC_MAP).bindPopup(label);
    });
  }

  if (!timeField){
    drawPoints(data);
    return;
  }

  const times = [...new Set(data.map(r=> r[timeField]))];
  let i=0;
  function drawFrame(k){
    window.AC_MAP.eachLayer(l=>{ if (l instanceof L.CircleMarker) window.AC_MAP.removeLayer(l); });
    const slice = data.filter(r=> r[timeField]===times[k]);
    drawPoints(slice);
  }
  drawFrame(0);

  if (times.length>1){
    if (window._geoTimer) clearInterval(window._geoTimer);
    window._geoTimer = setInterval(()=>{
      i = (i+1) % times.length;
      drawFrame(i);
    }, Math.max(350, dur));
  }
}


async function acRender(spec){
  AC_SPEC = spec;
  document.getElementById('ac_meta').textContent = acMeta(spec);

  if (['bar','line','pie','heatmap'].includes(spec.chart_type)){
    acHideAll();
    const pack = acToPlotly(spec);
    acAnimatePlotly('ac_plot', pack);
    return;
  }
  if (spec.chart_type === 'network'){
    acRenderNetwork(spec);
    return;
  }
  if (spec.chart_type === 'geo'){
    acRenderGeo(spec);
    return;
  }
}

async function acPlan(){
  const q = document.getElementById('ac_q').value.trim();
  document.getElementById('ac_meta').textContent='…';
  const d = await fetch('/autochart/plan', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({q})}).then(r=>r.json());
  if (!d.ok){ document.getElementById('ac_meta').textContent = 'Error: '+(d.error||''); return; }
  acRender(d.spec);
}

async function acSample(){
  document.getElementById('ac_meta').textContent='…';
  const kind = document.getElementById('ac_kind').value || 'bar';
  const d = await fetch('/autochart/sample', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({kind})
  }).then(r=>r.json());
  if (!d.ok){ document.getElementById('ac_meta').textContent = 'Error: '+(d.error||''); return; }
  acRender(d.spec);
}

document.getElementById('ac_plan').onclick = acPlan;
document.getElementById('ac_sample').onclick = acSample;
</script>

<script src="https://unpkg.com/cytoscape@3/dist/cytoscape.min.js"></script>
<script type="module">
  import { initLessonPlayer } from "/static/lesson_player.js";
  document.getElementById("lessonStart").onclick = () => initLessonPlayer({ useMermaid: true });
</script>

</body></html>
"""
  return HTMLResponse(page)

@router.get("/studio", response_class=HTMLResponse)
def studio_alias(request: Request, q: str = Query("", description="Ask the model")):
  return scholarviz(request, q)

@router.get("/", response_class=HTMLResponse)
def root_alias(request: Request):
  return scholarviz(request, q="")

def _rng(seed:int):
  r = random.Random()
  r.seed(seed if seed else random.SystemRandom().randint(1, 10**9))
  return r

def _shuffle_with_seed(xs, seed, do=False):
  xs = list(xs)
  if do:
    r = _rng(seed); r.shuffle(xs)
  return xs

def _uniq(xs):
  seen=set(); out=[]
  for x in xs:
    if x not in seen:
      seen.add(x); out.append(x)
  return out

def _try_imports():
  mod = {}
  try:
    import fitz
    mod['fitz'] = fitz
  except Exception:
    mod['fitz'] = None
  try:
    import pytesseract
    from PIL import Image
    mod['pytesseract'] = pytesseract
    mod['PIL_Image'] = Image
  except Exception:
    mod['pytesseract'] = None
    mod['PIL_Image'] = None
  return mod

_IM = _try_imports()

async def _fetch_pdf_bytes(url: str) -> bytes | None:
  try:
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
      r = await hx.get(url); r.raise_for_status()
      return r.content
  except Exception:
    return None

def _tok(s:str)->set:
  return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def _relevance_score(p:dict, must: set, nice: set) -> int:
  title = (p.get("title") or "")
  venue = ((p.get("primary_location") or {}).get("source") or {}).get("display_name","")
  inv = (p.get("abstract_inverted_index") or {})
  abstract = " ".join(inv.keys()) if isinstance(inv, dict) else ""
  bag = _tok(title + " " + venue + " " + abstract)
  score = 0
  score += 10*len(must & bag)
  score += 2*len(nice & bag)
  return score

async def _open_access_hits(q: str, per: int = 8, seed:int=0, diversify:bool=False) -> List[dict]:
  url = "https://api.openalex.org/works"
  now = datetime.date.today()
  start = f"{now.year-4}-01-01"

  must = _tok("data loss prevention saas")
  nice = _tok("policy engine classifier quarantine allow cloud security zero trust access control dlp")

  base = _normalize_for_oa(q) or [q]
  extra = [
    "data loss prevention SaaS",
    "cloud DLP policy engine",
    "SaaS security data exfiltration",
    "content classification policy quarantine"
  ]
  queries = []
  [queries.append(x) for x in (base[:2] + extra) if x not in queries]
  queries = _uniq(queries)
  queries = _shuffle_with_seed(queries, seed, diversify)

  async def _hit_once(hx, search: str, relax: bool) -> List[dict]:
    params = {
      "search": search,
      "filter": f"from_publication_date:{start},language:en" + (",primary_location.source.type:journal" if not relax else ""),
      "sort": "publication_date:desc",
      "per_page": max(per, 20)
    }
    r = await hx.get(url, params=params); r.raise_for_status()
    return r.json().get("results", []) or []

  results = []
  try:
    async with httpx.AsyncClient(timeout=16.0) as hx:
      for relax in (False, True):
        for q1 in queries:
          try:
            got = await _hit_once(hx, q1, relax=relax)
            if got: results.extend(got)
          except Exception:
            continue
        if results: break
  except Exception:
    results = []

  ranked = []
  for w in results:
    loc = (w.get("primary_location") or {})
    oa  = (w.get("open_access") or {})
    pdf_url = loc.get("pdf_url") or oa.get("oa_url") or ""
    if not pdf_url: continue
    ranked.append((_relevance_score(w, must, nice), w, pdf_url))

  ranked.sort(key=lambda t: -t[0])
  ranked = [t for t in ranked if t[0] >= 10] or ranked

  out = []
  for sc, w, pdf_url in ranked[:per]:
    loc = (w.get("primary_location") or {})
    src = (loc.get("source") or {})
    out.append({
      "title": w.get("title") or "(untitled)",
      "year": w.get("publication_year") or "",
      "venue": src.get("display_name"),
      "pdf": pdf_url,
      "url": loc.get("landing_page_url") or (w.get("open_access") or {}).get("oa_url") or w.get("id"),
      "authors": ", ".join(a.get("author",{}).get("display_name","") for a in (w.get("authorships") or [])[:4])
    })
  return out

def _split_blocks(txt: str) -> List[str]:
  txt = re.sub(r"\s+", " ", txt or "").strip()
  blocks = re.split(r"(?<=\.)\s{2,}|(?:\n\s*){2,}", txt)
  return [b.strip() for b in blocks if len(b.strip()) > 120]

def _fig_captions(raw: str) -> List[str]:
  caps = re.findall(r"(?:Figure|Fig\.?)\s*\d+[:\.\)]\s*[^\n\.]+(?:\.[^\n\.]+)?", raw, flags=re.I)
  return [c.strip() for c in caps][:40]

def _rank_snippets(q: str, snippets: List[str], k: int = 10, seed:int=0, diversify:bool=False) -> List[dict]:
  def _jitter(x, r, do):
    return x + (r.uniform(-1e-3, 1e-3) if do else 0.0)

  r = _rng(seed)
  try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vec = TfidfVectorizer(stop_words="english", max_features=8000)
    X = vec.fit_transform(snippets + [q])
    sims = (X[:-1] @ X[-1].T).toarray().ravel()
    if diversify:
      sims = np.array([_jitter(float(s), r, True) for s in sims])
    idx = np.argsort(-sims)[:k]
    return [{"i": int(i), "text": snippets[int(i)], "score": float(sims[int(i)])} for i in idx]
  except Exception:
    ql = set(re.findall(r"[a-z0-9]+", q.lower()))
    scored = []
    for i, s in enumerate(snippets):
      sl = set(re.findall(r"[a-z0-9]+", s.lower()))
      scored.append((i, len(ql & sl)))
    if diversify:
      scored = [(i, sc + r.random()*1e-3) for i, sc in scored]
    scored.sort(key=lambda t: -t[1])
    return [{"i": i, "text": snippets[i], "score": float(sc)} for i, sc in scored[:k]]

async def _fetch_pdf_text(url: str) -> str:
  try:
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
      r = await hx.get(url); r.raise_for_status()
      b = r.content
  except Exception:
    return ""
  txt = ""
  try:
    if PdfReader is not None:
      reader = PdfReader(io.BytesIO(b))
      for p in reader.pages[:20]: txt += (p.extract_text() or "") + "\n"
    elif _IM['fitz']:
      fitz = _IM['fitz']
      doc = fitz.open(stream=b, filetype="pdf")
      for i, page in enumerate(doc):
        if i >= 20: break
        txt += (page.get_text() or "") + "\n"
  except Exception:
    pass
  return txt

async def _fetch_pdf_figures(url: str, max_pages: int =12, max_imgs: int = 12) -> List[dict]:
  out: List[dict] = []
  if not _IM['fitz']:
    return out
  b = await _fetch_pdf_bytes(url)
  if not b: return out
  try:
    fitz = _IM['fitz']
    doc = fitz.open(stream=b, filetype="pdf")
    img_count = 0
    for i, page in enumerate(doc):
      if i >= max_pages or img_count >= max_imgs: break
      for img in page.get_images(full=True):
        if img_count >= max_imgs: break
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n >= 4:
          pix = fitz.Pixmap(fitz.csRGB, pix)
        png_b = pix.tobytes("png")
        png_b64 = "data:image/png;base64," + base64.b64encode(png_b).decode("ascii")
        text = (page.get_text() or "")
        cap = ""
        m = re.search(r"(Figure|Fig\.?)\s*\d+[:\.\)]\s*[^\n\.]+", text, flags=re.I)
        if m: cap = m.group(0).strip()
        ocr = ""
        if _IM['pytesseract'] and _IM['PIL_Image']:
          try:
            from PIL import Image as PILImage
            import io as _io
            ocr = _IM['pytesseract'].image_to_string(PILImage.open(_io.BytesIO(png_b)))[:300]
          except Exception:
            ocr = ""
        out.append({"page": i+1, "caption": cap, "png_b64": png_b64, "ocr": ocr})
        img_count += 1
  except Exception:
    pass
  return out

GROUNDED_RULES = (
  "Use provided snippets and figure captions/OCR to synthesize a small Mermaid flowchart (5–8 nodes, LR). "
  "Prefer structure suggested by figure captions. Include 2–4 dashed control nodes. "
  "Output JSON: {mermaid:string, summary:string, cites:number[]}. "
  "Each node/edge must be inferable from snippets/figures; cite indices in 'cites' and reference them inline as [#] in the summary. "
  "Use exact nouns from captions where reasonable. Defense-only. If insufficient evidence, return mermaid:'' and a brief reason."
)

@router.post("/diagram/grounded")
async def diagram_grounded(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  fast = bool((payload or {}).get("fast"))
  per = 3 if fast else 8
  n = int((payload or {}).get("n") or 1)
  seed = int((payload or {}).get("seed") or 0)
  diversify = bool((payload or {}).get("diversify") or False)
  temp = float((payload or {}).get("temp") or 0.2)

  if _risky(q): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)

  hits = await _open_access_hits(q, per=8, seed=seed, diversify=diversify)

  if not hits:
    mer, summ = _llm_mermaid_plan(q)
    if not mer:
      mer = _skeleton_from_text(q)
    if not _is_plausible_mermaid(mer):
      mer = _default_mermaid()
    return JSONResponse({
      "ok": True,
      "mermaid": mer,
      "summary": "(Fallback: no_oa_pdfs) " + (summ or "Synthesized without OA PDFs."),
      "cites": [],
      "snippets": [],
      "sources": []
    })

  texts = []
  for h in hits[:3]:
    t = await _fetch_pdf_text(h["pdf"])
    if t: texts.append((h, t))

  if not texts:
    mer, summ = _llm_mermaid_plan(q)
    if not mer:
      mer = _skeleton_from_text(q)
    if not _is_plausible_mermaid(mer):
      mer = _default_mermaid()
    return JSONResponse({
      "ok": True,
      "mermaid": mer,
      "summary": "(Fallback: no_oa_text) " + (summ or "Synthesized without extracted PDF text."),
      "cites": [],
      "snippets": [],
      "sources": hits
    })

  figs_all: List[dict] = []
  for h, _ in texts:
    try:
      figs = await _fetch_pdf_figures(h["pdf"])
    except Exception:
      figs = []
    if figs:
      for f in figs[:6 - len(figs_all)]:
        figs_all.append({"paper": h["title"], **f})
        if len(figs_all) >= 6: break
    if len(figs_all) >= 6: break

  pool = []
  for h, t in texts:
    caps = _fig_captions(t)
    blks = _split_blocks(t)
    pool.extend([f"[cap] {c}" for c in caps] + blks)

  if figs_all:
    for fi in figs_all:
      if fi.get("caption"):
        pool.insert(0, f"[figcap p{fi['page']}] {fi['caption']}")
      if fi.get("ocr"):
        pool.insert(0, f"[figocr p{fi['page']}] {fi['ocr']}")

  top = _rank_snippets(q, pool, k=10, seed=seed, diversify=diversify)
  bundle = {
    "query": q,
    "snippets": [{"idx": s["i"], "text": s["text"]} for s in top],
    "figures": [{"page": f["page"], "caption": f["caption"]} for f in figs_all]
  }

  try:
    def _one_variant(bundle, seed_offset=0):
      user = json.dumps(bundle, ensure_ascii=False)
      rules = GROUNDED_RULES + f" (variation_seed={seed+seed_offset})"
      res = gen_json(rules, user, model=MODEL, temperature=temp) or {}
      mer = _repair_mermaid((res or {}).get("mermaid") or "") or _ascii_to_mermaid((res.get("summary") or ""))
      return {
        "mermaid": mer,
        "summary": (res or {}).get("summary") or "",
        "cites": (res or {}).get("cites") or []
      }

    variants = []
    for i in range(max(1, n)):
      variants.append(_one_variant(bundle, seed_offset=i))

    best = next((v for v in variants if v["mermaid"]), variants[0])
    mer = best["mermaid"]
    summ = best["summary"]
    cites = best["cites"]

    ql = (q or "").lower()
    wants_dlp = ("data loss prevention" in ql or "dlp" in ql) and ("saas" in ql)
    describes_flow = ("->" in q) or (" to " in ql and "policy" in ql and ("quarantine" in ql or "allow" in ql))
    if not mer and (wants_dlp or describes_flow):
      mer = _diagram_dlp()
      summ = "(Fallback: dlp_seed) Seeded DLP pipeline rendered due to low-evidence OA results."

    if not mer:
      seed_text = " ".join(s.get("text","") for s in bundle["snippets"][:6]) or q
      mer = _skeleton_from_text(seed_text)
    if not _is_plausible_mermaid(mer):
      mer = _default_mermaid()

    return JSONResponse({
      "ok": True,
      "mermaid": mer,
      "summary": summ,
      "cites": cites,
      "snippets": bundle["snippets"],
      "figures": figs_all,
      "sources": hits,
      "alternatives": variants
    })
  except Exception as e:
    mer_fb, sum_fb = _llm_mermaid_plan(q, req_labels=None)
    if not mer_fb:
      mer_fb = _skeleton_from_text(q)
    if not _is_plausible_mermaid(mer_fb):
      mer_fb = _default_mermaid()
    srcs = await _scholar_search((_normalize_for_oa(q)[0] if _normalize_for_oa(q) else q), per=5)
    return JSONResponse({
      "ok": True,
      "mermaid": mer_fb,
      "summary": "(Fallback: grounded_llm_error) " + (sum_fb or "Synthesized from model priors."),
      "cites": [],
      "snippets": [],
      "figures": [],
      "sources": srcs,
      "fallback": "grounded_llm_error"
    })
