# trustviz/server/notebook_routes.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import json, traceback
from trustviz.notebook.runner import NotebookRunner
from trustviz.llm.gemini_client import gen_text

router = APIRouter(prefix="/notebook", tags=["notebook"])

@router.post("/run")
async def run_notebook(request: Request):
    try:
        body = await request.json()
        spec = body.get("spec") or {}

        # Build the validation cell WITHOUT f-string braces in comments.
        cell = "\n".join([
            "import json, networkx as nx",
            "ARTIFACTS['checks'] = {}",
            f"spec = json.loads('''{json.dumps(spec)}''')",
            "",
            "G = nx.DiGraph()",
            "for n in spec.get('nodes', []):",
            "    nid = n.get('id') or n.get('label')",
            "    if nid: G.add_node(nid)",
            "",
            "# Edges can be lists/tuples like ['A','B'] or dicts like {'src':'A','dst':'B'}",
            "for e in spec.get('edges', []):",
            "    if isinstance(e, (list, tuple)) and len(e) >= 2:",
            "        G.add_edge(e[0], e[1])",
            "    elif isinstance(e, dict):",
            "        src = e.get('src') or e.get('source')",
            "        dst = e.get('dst') or e.get('target')",
            "        if src and dst: G.add_edge(src, dst)",
            "",
            "checks = ARTIFACTS['checks']",
            "checks['node_count'] = G.number_of_nodes()",
            "checks['edge_count'] = G.number_of_edges()",
            "checks['is_dag'] = nx.is_directed_acyclic_graph(G)",
            "checks['isolated_nodes'] = [n for n in G.nodes() if G.degree(n) == 0]",
            "checks['defense_count'] = len(spec.get('defenses', []) or [])",
            "",
            "verdicts = []",
            "if not checks['is_dag']: verdicts.append('Flow contains cycles; remove back-edges.')",
            "if checks['defense_count'] == 0: verdicts.append('No defensive controls attached.')",
            "if checks['isolated_nodes']: verdicts.append('Isolated nodes: ' + json.dumps(checks['isolated_nodes'][:5]))",
            "ARTIFACTS['verdicts'] = verdicts or ['OK']",
        ])

        runner = NotebookRunner(timeout_s=6)
        try:
            res = runner.run([cell])
            return JSONResponse({"ok": True, "artifacts": res.artifacts or {}})
        finally:
            runner.stop()
    except Exception:
        return JSONResponse({"ok": False, "error": traceback.format_exc()}, status_code=500)

# trustviz/server/notebook_routes.py
@router.post("/explain")
async def explain_artifacts(request: Request):
    try:
        body = await request.json()
        artifacts = body.get("artifacts") or {}
        question  = (body.get("question") or "").strip()

        # NEW: handle empty / missing artifacts clearly
        if not artifacts or not isinstance(artifacts, dict) or not artifacts.get("checks"):
            msg = (
                "No validation results were produced.\n\n"
                "Tips:\n"
                "• Click “Validate in Notebook” first.\n"
                "• Ensure your spec has at least two nodes and one edge, e.g.\n"
                "  {\"nodes\":[{\"id\":\"A\"},{\"id\":\"B\"}], \"edges\":[[\"A\",\"B\"]], \"defenses\":[]}\n"
                "• If the output still shows {}, check the server logs for Notebook/kernel errors."
            )
            return JSONResponse({"ok": True, "text": msg})

        sys = "You explain validation results for a security process diagram. Be concise and actionable. Defense-only."
        usr = (
            "Validation artifacts (JSON):\n"
            + json.dumps(artifacts, indent=2)
            + (f"\n\nStudent asked: {question}\n" if question else "\n")
            + "Give a short explanation grounded only in these artifacts."
        )
        text = gen_text(sys, usr, model=None, temperature=0.2)
        return JSONResponse({"ok": True, "text": text})
    except Exception:
        return JSONResponse({"ok": False, "error": traceback.format_exc()}, status_code=500)
