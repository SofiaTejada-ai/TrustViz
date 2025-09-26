import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_roc(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_score)

    # Keep all ROC points for correct AUC; only sanitize thresholds for JSON.
    thr_list = []
    for t in thr:
        if np.isfinite(t):
            thr_list.append(float(t))
        else:
            thr_list.append(None)   # becomes JSON null

    return {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "thresholds": thr_list,
        "auc": float(auc(fpr, tpr)),
    }

def auc_close(a: float, b: float, tol: float = 0.01) -> bool:
    try:
        a = float(a); b = float(b)
    except Exception:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= tol
