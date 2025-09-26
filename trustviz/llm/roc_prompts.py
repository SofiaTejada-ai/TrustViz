
ROC_JSON_INSTRUCTIONS = """You are generating a ROC chart spec as strict JSON.
Return ONLY JSON with keys:
"title", "fpr", "tpr", "thresholds", "auc", "x_label", "y_label", "alt_text".
- fpr/tpr/thresholds: arrays of numbers
- auc: number
- labels: short strings
NO extra keys, NO comments, NO prose.
"""

def roc_user_prompt():
    return ("Create a ROC spec for binary classification. "
            "If unsure, approximate but keep values in [0,1].")
