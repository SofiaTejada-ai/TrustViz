# trustviz/server/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np

from trustviz.svl.chart_spec import RocSpec
from trustviz.svl.roc_verify import compute_roc, auc_close
from trustviz.ve.plotly_adapter import roc_figure_json

app = FastAPI(title="TrustViz")

@app.get("/")
def root():
    return {"ok": True, "service": "TrustViz", "routes": ["/", "/health", "/docs", "/roc"]}

@app.get("/health")
def health():
    return {"ok": True}

# --- Request model for /roc ---

class RocRequest(BaseModel):
    y_true: conlist(int, min_length=2)      # v2: min_length
    y_score: conlist(float, min_length=2)   # v2: min_length
    llm_spec: RocSpec | None = None
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    alt_text: str | None = None

@app.post("/roc")
def roc_endpoint(req: RocRequest):
    # 1) Recompute ROC from provided data (source of truth)
    y_true = np.array(req.y_true, dtype=int)
    y_score = np.array(req.y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise HTTPException(status_code=400, detail="y_true and y_score must have same length.")

    recomputed = compute_roc(y_true, y_score)

    # 2) If an LLM spec is provided, verify AUC agreement; otherwise we build a canonical spec
    if req.llm_spec is not None:
        llm_auc = req.llm_spec.auc
        if not auc_close(llm_auc, recomputed["auc"], tol=0.01):
            # Override incorrect numbers with recomputed truth
            final_spec = RocSpec(
                title=req.title or req.llm_spec.title,
                fpr=recomputed["fpr"],
                tpr=recomputed["tpr"],
                thresholds=recomputed["thresholds"],
                auc=recomputed["auc"],
                x_label=req.x_label or req.llm_spec.x_label,
                y_label=req.y_label or req.llm_spec.y_label,
                alt_text=req.alt_text or req.llm_spec.alt_text,
            )
            decision = "corrected_mismatch"
        else:
            # Keep the llm spec numbers, but we still trust-check they pass schema
            final_spec = RocSpec(**req.llm_spec.dict())
            decision = "accepted_llm"
    else:
        # Build canonical spec purely from recomputed truth
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
        decision = "computed_truth"

    # 3) Produce a safe Plotly JSON for rendering
    fig_json = roc_figure_json(final_spec.fpr, final_spec.tpr,
                               final_spec.title, final_spec.x_label, final_spec.y_label)

    return {
        "decision": decision,
        "spec": final_spec.dict(),
        "figure": fig_json
    }
