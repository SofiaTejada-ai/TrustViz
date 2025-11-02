# trustviz/server/notebook_routes.py — Notebook runner endpoints used by Studio
import json
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

# Server-side notebook sandbox runner
from trustviz.notebook.runner import NotebookRunner
# LLM explainer (defense-only + short)
from trustviz.llm.gemini_client import gen_json

router = APIRouter()

# ------------------------------ Small helpers ------------------------------

_TEMPLATES = [
    {
        "label": "Plot a function y = sin(x) on [-6.28, 6.28]",
        "code": """# Sample & plot a function (helpers already imported)
tv_sample_function('sin(x)', xmin=-6.283, xmax=6.283, n=800)
print("Plotted sin(x). Points:", ARTIFACTS['checks'].get('points'))
"""
    },
    {
        "label": "Compute and plot an ROC curve",
        "code": """# Give some fake scores
pos = [0.91, 0.88, 0.73, 0.65, 0.61, 0.60]
neg = [0.55, 0.52, 0.49, 0.40, 0.33, 0.21]
auc = tv_roc(pos, neg)
print("AUC:", auc)
"""
    },
    {
        "label": "Toposort a small DAG",
        "code": """edges = [('A','B'), ('B','C'), ('A','D')]
order = tv_toposort(edges)
print("Topological order:", order)
"""
    },
    {
        "label": "Shortest path in a directed graph",
        "code": """edges = [('A','B'), ('B','D'), ('A','C'), ('C','D')]
path = tv_shortest_path(edges, 'A', 'D')
print("Shortest path A→D:", path)
"""
    },
]

# One prologue cell that defines the helpers the UI mentions.
# NotebookRunner exposes a dict-like ARTIFACTS; we extend it here.
_PROLOGUE = r"""
import io, base64, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import numpy as np
except Exception:
    np = None

try:
    import networkx as nx
except Exception:
    nx = None

if 'ARTIFACTS' not in globals():
    ARTIFACTS = {}
ARTIFACTS.setdefault('images', [])
ARTIFACTS.setdefault('checks', {})
ARTIFACTS.setdefault('verdicts', [])

def _img_to_data_url(fig=None, dpi=160):
    f = fig or plt.gcf()
    buf = io.BytesIO()
    f.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('ascii')
    buf.close()
    plt.close(f)
    return data

def tv_show(fig=None):
    "Save a matplotlib figure into ARTIFACTS['images'] and close it."
    ARTIFACTS['images'].append(_img_to_data_url(fig))

def tv_sample_function(expr='sin(x)', xmin=-5.0, xmax=5.0, n=600):
    import math as m
    xs = []
    ys = []
    def E(x):
        env = {'sin':m.sin,'cos':m.cos,'tan':m.tan,'exp':m.exp,'log':m.log,'sqrt':m.sqrt,'pi':m.pi,'e':m.e,'x':x}
        return eval(expr.replace('^','**'), {'__builtins__':{}}, env)
    step = (xmax - xmin) / max(n-1, 1)
    for i in range(n):
        x = xmin + i*step
        try:
            y = float(E(x))
            if math.isfinite(y):
                xs.append(x); ys.append(y)
        except Exception:
            pass
    if len(xs) < 10:
        ARTIFACTS.setdefault('verdicts', []).append('Function sampling produced < 10 valid points.')
    fig = plt.figure()
    plt.plot(xs, ys)
    plt.title(f'y = {expr}')
    plt.xlabel('x'); plt.ylabel('y')
    tv_show(fig)
    ARTIFACTS['checks']['points'] = len(xs)

def tv_roc(pos, neg):
    A = None
    try:
        import numpy as _np
        from sklearn.metrics import roc_curve, auc
        y_true  = _np.array([1]*len(pos) + [0]*len(neg))
        y_score = _np.array(list(pos) + list(neg))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        A = float(auc(fpr, tpr))
        xs = fpr; ys = tpr
    except Exception:
        # Fallback AUC via pairwise comparisons
        P = list(pos); N = list(neg)
        total = len(P)*len(N)
        greater = sum(1 for p in P for n in N if p > n)
        equal   = sum(1 for p in P for n in N if p == n)
        A = (greater + 0.5*equal)/total if total else float('nan')
        # Naive monotone curve
        pts = sorted([(p,1) for p in P] + [(n,0) for n in N], key=lambda x: -x[0])
        tp=fp=0; X=[0.0]; Y=[0.0]; Pn=len(P); Nn=len(N)
        for _,y in pts:
            tp += (y==1); fp += (y==0)
            X.append(fp/max(Nn,1)); Y.append(tp/max(Pn,1))
        xs = X; ys = Y
    fig = plt.figure()
    plt.plot(xs, ys)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC (AUC={A:.3f})' if A==A else 'ROC')
    tv_show(fig)
    ARTIFACTS['checks']['auc'] = A
    return A

def tv_toposort(edges):
    if nx is None:
        ARTIFACTS.setdefault('verdicts', []).append('networkx not available.')
        return []
    G = nx.DiGraph()
    G.add_edges_from(edges)
    ok = nx.is_directed_acyclic_graph(G)
    if not ok:
        ARTIFACTS.setdefault('verdicts', []).append('Graph has cycles.')
    order = list(nx.topological_sort(G)) if ok else []
    ARTIFACTS['checks']['node_count'] = G.number_of_nodes()
    ARTIFACTS['checks']['edge_count'] = G.number_of_edges()
    return order

def tv_shortest_path(edges, src, dst):
    if nx is None:
        ARTIFACTS.setdefault('verdicts', []).append('networkx not available.')
        return []
    G = nx.DiGraph()
    G.add_edges_from(edges)
    try:
        path = nx.shortest_path(G, src, dst)
        ARTIFACTS['checks']['path_len'] = len(path)
        return path
    except Exception as e:
        ARTIFACTS.setdefault('verdicts', []).append(str(e))
        return []
"""

_EXPLAIN_RULES = (
    "You are explaining results from a safe, local notebook sandbox. "
    "Defense-only; never provide offensive instructions. "
    "Return a short paragraph (<=160 words) that: "
    "1) summarizes the artifacts the student got (mention keys like images/verdicts/checks/stdout if present), "
    "2) states 1–3 takeaways or next steps, "
    "3) uses plain language."
)

# ------------------------------ Endpoints -----------------------------------

@router.get("/notebook/templates")
def notebook_templates():
    # UI expects: { ok: true, templates: [{label, code}, ...] }
    return JSONResponse({"ok": True, "templates": _TEMPLATES})

@router.post("/notebook/run_cell")
def notebook_run_cell(payload: dict = Body(...)):
    code = (payload or {}).get("code", "")
    if not isinstance(code, str) or not code.strip():
        return JSONResponse({"ok": False, "error": "Empty code."}, status_code=400)

    runner = NotebookRunner(timeout_s=10)  # keep tight
    try:
        # We run the prologue once + the user's cell.
        cells = [_PROLOGUE, code]
        res = runner.run(cells)
        arts = res.artifacts or {}
        # Normalize keys so the UI can always print something useful.
        out = {
            "stdout": arts.get("stdout", ""),
            "images": arts.get("images", []),
            "checks": arts.get("checks", {}),
            "verdicts": arts.get("verdicts", []),
        }
        return JSONResponse({"ok": True, "artifacts": out})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)
    finally:
        try:
            runner.stop()
        except Exception:
            pass

@router.post("/notebook/explain_code")
def notebook_explain_code(payload: dict = Body(...)):
    artifacts = (payload or {}).get("artifacts") or {}
    q = (payload or {}).get("question", "") or ""
    try:
        user = (
            "Notebook artifacts JSON:\n" + json.dumps(artifacts) + "\n\n"
            "Student question/focus (optional): " + q
        )
        text = gen_json(_EXPLAIN_RULES, user)
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)
        return JSONResponse({"ok": True, "text": text})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{e}"}, status_code=500)
