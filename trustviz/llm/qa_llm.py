# trustviz/llm/qa_llm.py
from typing import Optional
from trustviz.llm.gemini_client import gen_text

SYSTEM = (
    "You are a patient cybersecurity instructor. "
    "Explain clearly, step by step, using correct terminology. "
    "Prefer concise paragraphs and short bullet lists. "
    "If the user asks for mitigation, give prioritized, practical actions. "
    "If they ask for risky or harmful guidance, refuse and explain why."
)

def get_explanation_markdown(question: str, model: Optional[str]) -> str:
    return gen_text(SYSTEM, question, model=model, temperature=0.3) or \
           "Sorry, I couldn't produce an explanation."
