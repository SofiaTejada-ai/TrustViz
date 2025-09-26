# trustviz/llm/roc_llm.py
#CALL THE MODEL

import json
from .openai_client import get_client
from .roc_prompts import ROC_JSON_INSTRUCTIONS, roc_user_prompt

def get_llm_roc_spec(model: str = "gpt-4o-mini") -> dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ROC_JSON_INSTRUCTIONS},
            {"role": "user", "content": roc_user_prompt()},
        ],
        temperature=0.7,
    )
    content = resp.choices[0].message.content
    # Some models wrap JSON in code fences; strip if present.
    content = content.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(content)
