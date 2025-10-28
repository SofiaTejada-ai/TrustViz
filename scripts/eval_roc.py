import json, requests, numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import os
URL = os.environ.get("TRUSTVIZ_BASE_URL", "http://127.0.0.1:8000") + "/roc/llm"


def run_once():
    X,y = make_classification(n_samples=400, n_features=8, weights=[0.7,0.3], flip_y=0.05, random_state=None)
    m = LogisticRegression(max_iter=1000).fit(X,y)
    p = m.predict_proba(X)[:,1].tolist()
    r = requests.post(URL, json={"y_true":y.tolist(), "y_score":p})
    r.raise_for_status(); j = r.json()
    return j["decision"], j["spec"]["auc"]

def main(N=50):
    acc = corr = inv = 0
    aucs = []
    for _ in range(N):
        d, a = run_once()
        aucs.append(a)
        acc += (d=="accepted_llm")
        corr += (d=="corrected_mismatch")
        inv += (d=="computed_truth" or d=="labels_only")
    print(f"runs={N}  accepted={acc}  corrected={corr}  fallback={inv}  mean_truth_auc={np.mean(aucs):.3f}")

if __name__ == "__main__":
    main()
