# trustviz/svl/roc_verify.py
import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_roc(y_true: np.ndarray, y_score: np.ndarray):
    """Recompute ROC from ground truth and scores (the source of truth)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
        "auc": float(auc(fpr, tpr)),
    }

def auc_close(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol
