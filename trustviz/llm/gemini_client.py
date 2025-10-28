# trustviz/llm/gemini_client.py
import os, json, re
from typing import Optional

# Try both modern and legacy Gemini SDK names
try:
    import google.generativeai as genai
except Exception as e:
    raise RuntimeError("google-generativeai is not installed: pip install google-generativeai") from e

_KEY = os.environ.get("GEMINI_API_KEY")
if not _KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment or .env")

genai.configure(api_key=_KEY)

def _make_model(model: Optional[str]):
    mdl = model or os.environ.get("TRUSTVIZ_LLM_MODEL", "gemini-2.5-pro")
    # Different SDK versions accept different arg names; try them in order.
    for arg in ("model_name", "name", None):
        try:
            if arg is None:
                return genai.GenerativeModel(mdl)          # newer versions
            return genai.GenerativeModel(**{arg: mdl})     # older versions
        except TypeError:
            continue
    # Last fallback:
    return genai.GenerativeModel(mdl)

def _coerce_json(s: str) -> dict:
    s = (s or "").strip()
    # Try to isolate a JSON object if extra text slipped in
    m = re.search(r"\{.*\}\s*$", s, re.S)
    if m:
        s = m.group(0)
    return json.loads(s) if s else {}

def gen_text(system: str,
             user: str,
             model: Optional[str] = None,
             temperature: float = 0.2) -> str:
    """Return plain text from Gemini given system+user prompts."""
    g = _make_model(model)
    prompt = f"{system.strip()}\n\n{user.strip()}"
    r = g.generate_content(prompt, generation_config={"temperature": temperature})
    if hasattr(r, "text") and r.text:
        return r.text.strip()
    # Newer API can return candidates
    try:
        return (r.candidates[0].content.parts[0].text or "").strip()
    except Exception:
        return ""

def gen_json(system: str,
             user: str,
             model: Optional[str] = None) -> dict:
    """
    Ask for JSON only. We still defensively extract the JSON in case
    the model adds framing text.
    """
    # Strong JSON instructions
    sys = (
        f"{system.strip()}\n"
        "Return ONLY JSON. No markdown fences. No extra prose."
    )
    txt = gen_text(sys, user, model=model, temperature=0.0)
    try:
        return _coerce_json(txt)
    except Exception:
        # Try a simple best-effort JSON block extraction
        try:
            return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception:
            return {}
