# trustviz/server/app.py
# trustviz/server/app.py (near the top after app = FastAPI(...))
import os

# 1) Imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from trustviz.svl.chart_spec import RocSpec
from trustviz.svl.roc_verify import compute_roc, auc_close
from trustviz.ve.plotly_adapter import roc_figure_json
from trustviz.llm.roc_llm import get_llm_roc_spec
from trustviz.server.middleware_policy import PolicyGuard


# 3) Create the app FIRST
app = FastAPI(title="TrustViz")
app.add_middleware(PolicyGuard)  # optional: only if you have the middleware

# 4) Simple routes
@app.get("/")
def root():
    return {"ok": True, "service": "TrustViz", "routes": ["/", "/health", "/docs", "/roc", "/roc/llm"]}


@app.get("/debug/env")
def debug_env():
    return {"has_openai_key": bool(os.environ.get("OPENAI_API_KEY"))}

@app.get("/health")
def health():
    return {"ok": True}

# 5) Models used by routes
class RocRequest(BaseModel):
    y_true: conlist(int, min_length=2)
    y_score: conlist(float, min_length=2)
    llm_spec: RocSpec | None = None
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    alt_text: str | None = None

# 6) Existing /roc route (if you have it)
@app.post("/roc")
def roc_endpoint(req: RocRequest):
    y_true = np.array(req.y_true, dtype=int)
    y_score = np.array(req.y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise HTTPException(status_code=400, detail="y_true and y_score must have same length.")
    recomputed = compute_roc(y_true, y_score)
    final_spec = RocSpec(
        title=req.title or "ROC Curve",
        fpr=recomputed["fpr"],
        tpr=recomputed["tpr"],
        thresholds=recomputed["thresholds"],
        auc=recomputed["auc"],
        x_label=req.x_label or "False Positive Rate",
        y_label=req.y_label or "True Positive Rate",
        alt_text=req.alt_text or "ROC curve showing model trade-off.",
    )
    fig_json = roc_figure_json(final_spec.fpr, final_spec.tpr, final_spec.title, final_spec.x_label, final_spec.y_label)
    return {"decision": "computed_truth", "spec": final_spec.dict(), "figure": fig_json}

# 7) LLM route — AFTER app is defined
@app.post("/roc/llm")
def roc_from_llm(y_true: conlist(int, min_length=2), y_score: conlist(float, min_length=2)):
    try:
        raw = get_llm_roc_spec()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    try:
        llm_spec = RocSpec(**raw)
    except Exception:
        llm_spec = None

    y_true_np = np.array(y_true, dtype=int)
    y_score_np = np.array(y_score, dtype=float)
    if y_true_np.shape != y_score_np.shape:
        raise HTTPException(status_code=400, detail="y_true and y_score must have same length.")

    recomputed = compute_roc(y_true_np, y_score_np)

    if llm_spec and auc_close(llm_spec.auc, recomputed["auc"], tol=0.01):
        final_spec = llm_spec
        decision = "accepted_llm"
    else:
        final_spec = RocSpec(
            title=(llm_spec.title if llm_spec else "ROC Curve"),
            fpr=recomputed["fpr"],
            tpr=recomputed["tpr"],
            thresholds=recomputed["thresholds"],
            auc=recomputed["auc"],
            x_label=(llm_spec.x_label if llm_spec else "False Positive Rate"),
            y_label=(llm_spec.y_label if llm_spec else "True Positive Rate"),
            alt_text=(llm_spec.alt_text if llm_spec else "ROC curve showing model trade-off."),
        )
        decision = "corrected_mismatch" if llm_spec else "computed_truth"

    fig_json = roc_figure_json(final_spec.fpr, final_spec.tpr, final_spec.title, final_spec.x_label, final_spec.y_label)
    return {"decision": decision, "spec": final_spec.dict(), "figure": fig_json}
