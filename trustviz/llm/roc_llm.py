# trustviz/llm/roc_llm.py
import os, json, re
from typing import Any, Dict, List
from openai import OpenAI

def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

def _model() -> str:
    return os.environ.get("TRUSTVIZ_LLM_MODEL", "gpt-4o-mini")

def _strict_json(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("empty LLM response")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*(.+?)\s*```", text, flags=re.S | re.I)
    if m:
        return json.loads(m.group(1))
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return json.loads(text[i:j+1])
    raise ValueError("no valid JSON object found in LLM response")

def _chat_json(messages: List[Dict[str,str]]) -> Dict[str, Any]:
    """
    Use Chat Completions with JSON mode (response_format) to force valid JSON.
    Works across 1.x SDKs.
    """
    client = _client()
    resp = client.chat.completions.create(
        model=_model(),
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},  # JSON mode
    )
    txt = resp.choices[0].message.content
    return _strict_json(txt)

# --------- Public helpers ---------

def get_llm_roc_spec() -> Dict[str, Any]:
    prompt = (
        "Output ONLY a JSON object with keys: "
        "title, fpr, tpr, thresholds, auc, x_label, y_label, alt_text. "
        "Constraints: fpr/tpr arrays are nondecreasing in [0,1] and same length>=2; "
        "thresholds in [0,1] strictly decreasing length>=2; auc in [0,1]. "
        "No text outside the JSON."
    )
    messages = [
        {"role":"system","content":"You are a calculator that returns STRICT JSON only."},
        {"role":"user","content":prompt},
    ]
    return _chat_json(messages)

def get_llm_roc_spec_from_data(y_true: List[int], y_score: List[float]) -> Dict[str, Any]:
    # keep prompt small if huge
    if len(y_true) > 200:
        step = max(1, len(y_true)//200)
        y_true  = y_true[::step]
        y_score = y_score[::step]
    payload = json.dumps({"y_true": y_true, "y_score": y_score})
    system = (
        "You are a calculator. Compute a ROC curve from provided y_true (0/1) and y_score ([0,1]). "
        "Return ONLY valid JSON with fields: title, fpr, tpr, thresholds, auc, x_label, y_label, alt_text. "
        "Constraints: fpr/tpr ∈[0,1] nondecreasing and same length; thresholds ∈[0,1] strictly decreasing; "
        "auc ∈[0,1]. No prose outside JSON."
    )
    user = f"Compute ROC spec strictly from this data:\n{payload}"
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":user},
    ]
    return _chat_json(messages)
