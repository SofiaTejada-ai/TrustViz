# trustviz/llm/diagram_from_papers.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import hashlib, json
from trustviz.llm.gemini_client import gen_json  # your existing Gemini JSON helper

# We return your existing graph spec, e.g., nodes/edges suitable for verify_kg + cyto adapter
DiagramSpec = Dict[str, Any]

SYSTEM_INSTRUCTIONS = """You are a cybersecurity research assistant.
Given a user question, a set of paper abstracts, and (optionally) figures extracted from the papers,
(1) decide if a diagram meaningfully helps answer the question; 
(2) if yes, propose ONE clear, original diagram that captures the key mechanics or flow; 
(3) output ONLY valid JSON per the schema.

Prefer: attack chains, data-flow, kill-chains, threat model graphs, or pipeline block diagrams.
If figures are provided, integrate their *ideas* (do NOT copy), and cite paper titles in 'sources'.
If a diagram would be noise, set "make_diagram": false with a short rationale.
"""

USER_TEMPLATE = """Question: {question}

Candidate papers (title :: abstract or venue/year):
{papers_list}

Extracted figure snippets (base64 pngs + captions may be present). Use them for *inspiration*, not duplication.
Number of figures: {nfigs}

Return JSON with keys:
- make_diagram: boolean
- rationale: short string
- diagram: (present only if make_diagram) {{
    "title": str,
    "type": "attack-chain" | "dataflow" | "threat-model" | "pipeline" | "state-machine",
    "nodes": [{{"id": str, "label": str, "kind": str}}],
    "edges": [{{"source": str, "target": str, "label": str}}],
    "notes": str
  }}
- sources: [{{"title": str, "year": int, "source": str}}]
"""

def _papers_to_bullets(papers) -> str:
    out = []
    for p in papers:
        venue_year = f"{p.venue or ''} {p.year or ''}".strip()
        out.append(f"- {p.title} :: {venue_year}\n  { (p.abstract or '')[:400] }")
    return "\n".join(out)

async def decide_and_build(question: str, papers, figures) -> Dict[str, Any]:
    user = USER_TEMPLATE.format(
        question=question.strip(),
        papers_list=_papers_to_bullets(papers),
        nfigs=len(figures)
    )

    imgs_payload = []
    for f in figures[:6]:
        imgs_payload.append({"image_b64": f.image_b64, "mime_type": "image/png", "caption": f.caption or ""})

    result = await gen_json(
        system=SYSTEM_INSTRUCTIONS,
        user=user,
        images=imgs_payload,          # your gemini client should accept multimodal JSON
        response_schema="strict_json" # if your helper supports a mode flag
    )
    # Expecting make_diagram bool; else fall back
    return result
