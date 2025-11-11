# trustviz/server/lesson_routes.py
import os, json, re
from typing import Dict, List
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from trustviz.llm.gemini_client import gen_json  # reuse your existing helper

router = APIRouter(tags=["lesson"])

# ----- Shared schema used by both endpoints ----------------------------------
LESSON_SCHEMA = {
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "initial": {
      "type": "object",
      "properties": {
        "nodes": {"type": "array", "items": {"type": "object"}},
        "edges": {"type": "array", "items": {"type": "object"}}
      }
    },
    "frames": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "note": {"type": "string"},
          "adds": {"type": "object", "properties": {
            "nodes": {"type": "array", "items": {"type": "object"}},
            "edges": {"type": "array", "items": {"type": "object"}}
          }},
          "highlight": {"type": "object", "properties": {
            "nodes": {"type": "array", "items": {"type": "string"}},
            "edges": {"type": "array", "items": {"type": "string"}}
          }},
          "fade": {"type": "object", "properties": {
            "nodes": {"type": "array", "items": {"type": "string"}},
            "edges": {"type": "array", "items": {"type": "string"}}
          }},
          "quiz": {"type": "object", "properties": {
            "prompt": {"type": "string"},
            "choices": {"type": "array", "items": {"type": "string"}},
            "answerIndex": {"type": "integer"}
          }}
        },
        "required": []
      }
    }
  },
  "required": ["title", "frames"]
}

LESSON_PROMPT_TMPL = """You are TrustViz LessonWriter.
Return ONLY JSON matching this schema (no prose). Teach step-by-step with short frames.

Topic: {topic}

Rules:
- Each frame should either add nodes/edges, highlight a path, or fade earlier context.
- Keep 6–10 frames max; concise notes (<=40 words).
- Use ids for nodes/edges so animations can match between frames.
- Use 'quiz' occasionally (multiple choice) with answerIndex 0-based.
Schema (for reference): {schema}
"""

@router.post("/lesson/plan")
async def lesson_plan(payload: Dict = Body(...)):
    """LLM-only, topic → LessonSpec (fast)."""
    topic = (payload or {}).get("topic", "").strip()
    if not topic:
        return JSONResponse({"error": "Missing 'topic'."}, status_code=400)
    prompt = LESSON_PROMPT_TMPL.format(topic=topic, schema=json.dumps(LESSON_SCHEMA))
    spec = gen_json(
        prompt,  # system
        topic,  # user
        model=os.environ.get("TRUSTVIZ_LLM_MODEL", "gemini-2.5-pro")
    )
    # Minimal guard: ensure frames list exists
    if not isinstance(spec, dict) or not isinstance(spec.get("frames"), list):
        spec = {"title":"Lesson", "initial":{}, "frames":[{"title":"Start","note":"(no frames)"}]}
    return JSONResponse(spec)

# ----- Deterministic: build LessonSpec directly from the current Mermaid -----

_NODE_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*[\[\(]([^\]\)]+)[\]\)]", re.M)
_EDGE_LINE = re.compile(r"\b([A-Za-z0-9_]+)\s*-\.*>\s*([A-Za-z0-9_]+)")

def _extract_nodes_edges_from_mermaid(mer: str):
    nodes: List[dict] = []
    edges: List[dict] = []
    seen = set()

    for m in _NODE_LINE.finditer(mer or ""):
        nid, label = m.group(1), m.group(2).strip()
        if nid not in seen:
            seen.add(nid)
            nodes.append({"id": nid, "label": label})

    seen_e = set()
    for m in _EDGE_LINE.finditer(mer or ""):
        a, b = m.group(1), m.group(2)
        eid = f"e_{a}_{b}"
        if eid not in seen_e:
            seen_e.add(eid)
            edges.append({"id": eid, "source": a, "target": b})
    return nodes, edges

@router.post("/lesson/plan_from_mermaid")
async def lesson_from_mermaid(payload: dict = Body(...)):
    """
    Deterministic animation over an already-rendered diagram.
    Takes Mermaid text, extracts nodes/edges, reveals nodes then edges in simple frames.
    """
    mer = (payload or {}).get("mermaid","")
    if not mer.strip():
        return JSONResponse({"error":"Missing mermaid"}, status_code=400)

    nodes, edges = _extract_nodes_edges_from_mermaid(mer)

    # initial: first two nodes if present
    initial_nodes = nodes[:2]
    initial = {"nodes": initial_nodes, "edges": []}
    have = {n["id"] for n in initial_nodes}

    frames: List[dict] = []

    # add remaining nodes
    for n in nodes[2:]:
        frames.append({
            "title": f"Add {n['label']}",
            "note": f"Introduce {n['label']}.",
            "adds": {"nodes":[n]},
            "highlight": {"nodes":[n["id"]]}
        })
        have.add(n["id"])

    # add edges in small batches
    batch = []
    def flush():
        nonlocal batch, frames
        if batch:
            frames.append({
                "title":"Connect steps",
                "note":"Add relationships for this stage.",
                "adds":{"edges": batch},
                "highlight":{"edges":[e["id"] for e in batch]}
            })
            batch = []

    for e in edges:
        if e["source"] in have and e["target"] in have:
            batch.append(e)
            if len(batch) >= 2:
                flush()
    flush()

    return JSONResponse({
        "title": "Animated walkthrough",
        "initial": initial,
        "frames": frames or [{"title":"Diagram loaded","note":"No steps to animate (single node)."}]
    })
