# trustviz/llm/kg_llm.py
from typing import Dict, Any, List, Optional
import json, re
from trustviz.llm.gemini_client import gen_json

ALLOWED_NODE_TYPES = {"host", "ip", "user", "domain", "process", "file", "hash", "cve"}
ALLOWED_REL_TYPES  = {
    "communicates_with", "resolves_to", "authenticates_as", "spawns",
    "writes", "reads", "connects_to", "downloads_from", "drops",
    "exploits", "observed_in", "related_to"
}

SYSTEM = (
    "You extract a small cybersecurity knowledge graph from text. "
    "Return ONLY JSON with keys: title, alt_text, nodes, edges. "
    "nodes: [{id, type, label}], edges: [{src, dst, rel}]. "
    f"Constraints: id is short slug (a-z0-9_-); type in {sorted(ALLOWED_NODE_TYPES)}; rel in {sorted(ALLOWED_REL_TYPES)}. "
    "Prefer 5–20 nodes. Do not invent facts beyond the text. No code fences."
)

_EXPLAIN_RULES = (
    "You explain cybersecurity knowledge graphs for students. "
    "Return ONLY JSON with keys: intro, outro. "
    "Each value is 1–3 sentences, concise, neutral, and defense-only. "
    "No code fences. No content outside JSON."
)

def get_llm_kg_walkthrough(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    spec = get_llm_kg_spec_from_text(text, model=model)

    labels = [n.get("label","") for n in spec.get("nodes", []) if isinstance(n, dict) and n.get("label")]

    intro = "This graph maps the key entities and their relationships, highlighting paths defenders monitor."
    outro = "Use the relationships to find chokepoints for detection and least-privilege; no operational steps are included."

    if model:
        try:
            user = (
                f"User prompt: {text}\n"
                f"Nodes: {[n.get('label') for n in spec.get('nodes',[])]}\n"
                f"Relations: {[e.get('rel') for e in spec.get('edges',[])]}\n"
                "Write the two paragraphs based only on this graph."
            )
            data = gen_json(_EXPLAIN_RULES, user, model=model) or {}
            if isinstance(data.get("intro"), str) and data["intro"].strip():
                intro = data["intro"].strip()
            if isinstance(data.get("outro"), str) and data["outro"].strip():
                outro = data["outro"].strip()
        except Exception:
            pass

    return {
        "segments": [
            {"type": "text", "content": intro},
            {"type": "viz",  "mode": "graph", "kg": spec},
            {"type": "text", "content": outro},
        ],
        "labels": labels,
    }

def get_llm_kg_spec_from_text(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    user_prompt = f"""Text:
{text}

Return JSON only. Example schema:
{{
  "title": "Investigation Graph",
  "alt_text": "Entities and relations from the report.",
  "nodes": [{{"id":"host-a","type":"host","label":"Host A"}}, {{"id":"10-0-0-5","type":"ip","label":"10.0.0.5"}}],
  "edges": [{{"src":"host-a","dst":"10-0-0-5","rel":"communicates_with"}}]
}}

Rules:
- Node types allowed: {sorted(ALLOWED_NODE_TYPES)}
- Edge rel types allowed: {sorted(ALLOWED_REL_TYPES)}
- Keep ≤ 20 nodes. Use concise, recognizable labels.
"""
    try:
        data = gen_json(SYSTEM, user_prompt, model=model) or {}
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    return _postprocess_ids(data)

def _slugify(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s or "n"

def _postprocess_ids(data: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = list(data.get("nodes", []) or [])
    edges: List[Dict[str, Any]] = list(data.get("edges", []) or [])

    seen = set()
    for n in nodes:
        nid = _slugify(n.get("id") or n.get("label", "n"))
        base = nid or "n"; i = 1
        while nid in seen:
            i += 1; nid = f"{base}-{i}"
        seen.add(nid)
        n["id"] = nid
        ntype = (n.get("type") or "host").lower()
        if ntype not in ALLOWED_NODE_TYPES:
            ntype = "host"
        n["type"] = ntype
        if not isinstance(n.get("label"), str) or not n["label"].strip():
            n["label"] = n["id"]

    idset = {n["id"] for n in nodes}
    for e in edges:
        e["src"] = _slugify(e.get("src", ""))
        e["dst"] = _slugify(e.get("dst", ""))
        if e["src"] not in idset:
            for n in nodes:
                if _slugify(n.get("label","")) == e["src"]:
                    e["src"] = n["id"]; break
        if e["dst"] not in idset:
            for n in nodes:
                if _slugify(n.get("label","")) == e["dst"]:
                    e["dst"] = n["id"]; break
        rel = (e.get("rel") or "related_to").lower()
        if rel not in ALLOWED_REL_TYPES:
            rel = "related_to"
        e["rel"] = rel

    data["nodes"] = nodes
    data["edges"] = edges
    data.setdefault("title", "Investigation Graph")
    data.setdefault("alt_text", "Entities and relations extracted from text.")
    return data
