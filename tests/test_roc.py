def test_auc_reasonable_range():
    from trustviz.svl.roc_verify import compute_roc
    import numpy as np
    y_true = np.array([0,0,1,1,1,0,1,0,1,0])
    y_score = np.array([0.1,0.3,0.9,0.8,0.7,0.2,0.65,0.4,0.55,0.05])
    r = compute_roc(y_true, y_score)
    assert 0.5 <= r["auc"] <= 1.0      # allow perfect separation
    # thresholds should be finite numbers or None (if +inf sanitized)
    assert all((t is None) or isinstance(t, float) for t in r["thresholds"])
    assert len(r["fpr"]) == len(r["tpr"]) == len(r["thresholds"])