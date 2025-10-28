# trustviz/llm/diagram_llm.py
# Two short paragraphs + a concrete caption that names what’s on the diagram.

from typing import Optional
from trustviz.llm.gemini_client import gen_json

_EXPLAIN_RULES = (
    "You write student-friendly EXPLANATIONS for a security process diagram. "
    "Return ONLY JSON with keys: intro, caption, outro. "
    "intro: 1–2 sentences of context. "
    "caption: 2–5 short sentences that explicitly mention 2–5 node labels and 2–5 edges/arrows and how the listed DEFENSIVE controls mitigate specific stages. "
    "outro: 1–2 sentences on how to use the diagram defensively. "
    "Always be defense-only. No code fences. No extra keys."
)

def get_diagram_walkthrough(prompt: str, mermaid_src: str, model: Optional[str]) -> dict:
    intro  = "This diagram shows the stages of a security process and where defenders add controls."
    caption = "Follow the arrows from left to right: each box is a stage, dotted arrows point to the control that mitigates that stage."
    outro  = "Use the mapping of stages to controls to review coverage and add missing guardrails."

    if model:
        try:
            user = (
                f"Student asked: {prompt}\n\n"
                "Here is the Mermaid diagram source you must comment on:\n"
                f"{mermaid_src}\n\n"
                "Extract the stage labels and any control names from the diagram text and use them in the caption."
            )
            data = gen_json(_EXPLAIN_RULES, user, model=model) or {}
            if isinstance(data, dict):
                if isinstance(data.get("intro"), str) and data["intro"].strip():
                    intro = data["intro"].strip()
                if isinstance(data.get("caption"), str) and data["caption"].strip():
                    caption = data["caption"].strip()
                if isinstance(data.get("outro"), str) and data["outro"].strip():
                    outro = data["outro"].strip()
        except Exception:
            pass

    # Return as three segments so the UI can show a paragraph, the diagram, then a rich caption.
    return {
        "segments": [
            {"type": "text", "content": intro},
            {"type": "viz",  "mode": "diagram", "mermaid": mermaid_src},
            {"type": "text", "content": caption},
            {"type": "text", "content": outro},
        ],
        "labels": [],  # we aren’t bold-highlighting terms for diagrams yet
    }
