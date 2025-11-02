# trustviz/server/studio_routes.py — ScholarViz (compact + sources + fixed ask)

import os, re, io, json, html, base64, datetime
from typing import List
from fastapi import APIRouter, Body, UploadFile, File, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pypdf import PdfReader
import httpx

from trustviz.llm.gemini_client import gen_text, gen_json
from trustviz.notebook.runner import NotebookRunner

router = APIRouter()
MODEL = os.environ.get("TRUSTVIZ_LLM_MODEL", "gemini-2.5-pro")

# ---------- safety ----------
_DENY = [
  r"\b(ddos|dos)\b", r"\b(botnet|c2|command\s*and\s*control)\b", r"\b(keylogger|rootkit|rat)\b",
  r"\bzero[-\s]?day\b", r"\bexploit\s+code\b", r"\bcredential\s+stuffing\b", r"\bphishing\s+kit\b",
  r"\bpassword\s*cracker\b", r"\bport\s*scan\s*(script|code)\b", r"\bsql\s*injection\s*(payload|exploit)\b",
  r"\bbypass\s+mfa\b", r"\breal\s+bank\s+site\b", r"\bhow\s+to\s+(hack|bypass)\b", r"\bbypass\b", r"\bhack(?:ing)?\b"
]
def _risky(s:str)->bool: return any(re.search(p,(s or "").lower()) for p in _DENY)

_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]()_:.-> \n|/%'\",+&?-")
def _sanitize_mermaid(s:str)->str: return "".join(ch for ch in (s or "") if ch in _ALLOWED)

def _default_mermaid()->str:
  return ("flowchart LR\n"
          "  Q[Question] --> D[High-level Diagram]\n"
          "  D --> K[Key Controls]\n"
          "  K --> O[Outcomes]\n")

# ---------- sources (OpenAlex) ----------
async def _scholar_search(q:str, per:int=5)->List[dict]:
  now = datetime.date.today()
  start = f"{now.year-3}-01-01"
  url = "https://api.openalex.org/works"
  params = {
    "search": q,
    "filter": f"from_publication_date:{start},language:en,primary_location.source.type:journal",
    "sort": "publication_date:desc",
    "per_page": per
  }
  try:
    async with httpx.AsyncClient(timeout=12.0) as hx:
      r = await hx.get(url, params=params)
      r.raise_for_status()
      data = r.json().get("results", [])
  except Exception:
    return []
  out = []
  for w in data:
    out.append({
      "title": w.get("title") or "(untitled)",
      "year": (w.get("publication_year") or ""),
      "venue": (w.get("primary_location",{}).get("source",{}) or {}).get("display_name"),
      "url": (w.get("primary_location",{}) or {}).get("landing_page_url") or w.get("open_access",{}).get("oa_url") or w.get("id"),
      "authors": ", ".join(a.get("author",{}).get("display_name","") for a in (w.get("authorships") or [])[:4])
    })
  return out

def _labels_from_mermaid(src:str)->List[str]:
  labs, seen = [], set()
  for m in re.finditer(r"\b[A-Za-z0-9_]+\s*[\[\(]([^\]\)]+)[\]\)]", src or ""):
    t = m.group(1).strip()
    if t and t not in seen: seen.add(t); labs.append(t)
  return labs[:20]

# ---------- special diagrams ----------
def _diagram_cia()->str:
  return _sanitize_mermaid("""flowchart LR
C[Confidentiality] --> P1[Access Control]
I[Integrity] --> P2[Checksums/Signing]
A[Availability] --> P3[Redundancy/Backups]
C -.-> DC1[Encryption at Rest/In Transit]
I -.-> DC2[Hash, Code Signing, WORM]
A -.-> DC3[Failover, DDoS Mitigation]
""")

def _diagram_zerotrust()->str:
  return _sanitize_mermaid("""flowchart LR
R[Request] --> P[Policy: Identity + Device + Context]
P --> G[Grant Least Privilege]
G --> V[Continuous Verification]
V --> Rv[Re-evaluate/Block]
P -.-> M1[MFA/WebAuthn]
P -.-> M2[Device Posture]
G -.-> M3[Segmentation]
""")

# ---------- LLM rules ----------
ASK_RULES = "Defense-only; <=160 words; use exact diagram labels if helpful; no offensive steps."
PLAN_RULES = (
  "Return JSON {make_diagram:boolean, mermaid?:string, summary:string}. "
  "If make_diagram=true, provide a small Mermaid flowchart (5–8 nodes L->R) and 2–4 dashed controls. "
  "Respect any 'Required labels:' text by including those labels verbatim as node titles."
)
DOC_RULES = (
  "Summarize like for a security student: 2-sentence overview; 3 key bullets; 2 defender takeaways. "
  "If a question is provided, answer briefly at the end."
)

# ---------- ingest store ----------
_INGEST = {"text":"", "images":[]}

# ===================== API =====================

@router.post("/ask")
async def ask(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  if _risky(q): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  try:
    # sources to display beside the answer
    sources = await _scholar_search(q, per=5)
    a = gen_text("Answer for a cybersecurity audience. Cite sources by [#] inline where useful.", q, model=MODEL) or ""
    return JSONResponse({"ok":True, "answer":a.strip(), "sources":sources})
  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)

@router.post("/diagram/plan")
async def diagram_plan(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  if _risky(q): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)

  # hard-match topics so labels appear exactly when requested
  ql = q.lower()
  if "cia triad" in ql or ("confidentiality" in ql and "integrity" in ql and "availability" in ql):
    mer = _diagram_cia()
    srcs = await _scholar_search("CIA triad confidentiality integrity availability security controls", per=5)
    return JSONResponse({"ok":True,"make_diagram":True,"mermaid":mer,"summary":"CIA triad with concrete controls.","sources":srcs})

  if "zero trust" in ql:
    mer = _diagram_zerotrust()
    srcs = await _scholar_search("zero trust architecture policy identity device context continuous verification", per=5)
    return JSONResponse({"ok":True,"make_diagram":True,"mermaid":mer,"summary":"Zero Trust decision path.","sources":srcs})

  try:
    # soft LLM plan that honors required labels
    req_labels = []
    m = re.search(r"required\s+labels\s*:\s*(.+)$", q, re.I)
    if m: req_labels = [t.strip() for t in re.split(r"[,/;]| and ", m.group(1)) if t.strip()]
    user = ("If labels are provided below, include them exactly as node titles.\n"
            + ("Required labels: " + ", ".join(req_labels) if req_labels else "Required labels:"))
    res = gen_json(PLAN_RULES, (q+"\n\n"+user), model=MODEL)
    make = bool(isinstance(res, dict) and res.get("make_diagram"))
    mer = _sanitize_mermaid((res or {}).get("mermaid") or ( _default_mermaid() if make else "")) if make else None
    summ = (res or {}).get("summary") or ""
    srcs = await _scholar_search(q, per=5)
    return JSONResponse({"ok":True,"make_diagram":make,"mermaid":mer,"summary":summ,"sources":srcs})
  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)

@router.post("/diagram/ask")
def diagram_ask(payload:dict=Body(...)):
  mer = (payload or {}).get("mermaid") or ""
  q = (payload or {}).get("q") or ""
  if not mer.strip(): return JSONResponse({"ok":False,"error":"No diagram loaded. Plan a diagram first."}, status_code=400)
  if _risky(q) or _risky(mer): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  labels = ", ".join(_labels_from_mermaid(mer))
  try:
    user = "Diagram:\n" + mer + "\n\nYou may reference labels: " + labels + "\n\nQuestion:\n" + q
    a = gen_text(ASK_RULES, user, model=MODEL) or ""
    return JSONResponse({"ok":True,"answer":a.strip()})
  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)

# ---------- notebook & docs ----------
@router.get("/notebook/templates")
def notebook_templates():
  tpls = [
    {"id":"cycle_check","label":"Graph sanity (cycles/isolates)","code":
     "import json, networkx as nx\nARTIFACTS['checks']={}\nG=nx.DiGraph()\n"
     "for n in spec.get('nodes',[]): G.add_node(n.get('id') or n.get('label'))\n"
     "for e in spec.get('edges',[]):\n"
     "    a,b=(e[0],e[1]) if isinstance(e,(list,tuple)) else (e.get('source'),e.get('target'))\n"
     "    if a and b: G.add_edge(a,b)\n"
     "ARTIFACTS['checks']['node_count']=G.number_of_nodes()\n"
     "ARTIFACTS['checks']['edge_count']=G.number_of_edges()\n"
     "ARTIFACTS['checks']['is_dag']=nx.is_directed_acyclic_graph(G)\n"
     "ARTIFACTS['checks']['isolated']=[n for n in G.nodes() if G.degree(n)==0]\n"
     "ARTIFACTS['verdicts']=(['OK'] if ARTIFACTS['checks']['is_dag'] and not ARTIFACTS['checks']['isolated'] else [])\n"
     "if not ARTIFACTS['checks']['is_dag']: ARTIFACTS.setdefault('verdicts',[]).append('Has cycle(s)')\n"
     "if ARTIFACTS['checks']['isolated']: ARTIFACTS.setdefault('verdicts',[]).append('Isolated nodes present')\n"},
    {"id":"sources_to_table","label":"Show sources as a table","code":
     "import pandas as pd\nARTIFACTS['sources_table']=pd.DataFrame(SOURCES).to_dict(orient='records')"}
  ]
  return JSONResponse({"ok":True,"templates":tpls})

@router.post("/notebook/run")
def notebook_run(payload:dict=Body(...)):
  code = (payload or {}).get("code") or ""
  spec = (payload or {}).get("spec") or {}
  sources = (payload or {}).get("sources") or []
  if _risky(json.dumps(spec)): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  nb = NotebookRunner(timeout_s=10)
  try:
    pre  = "ARTIFACTS={}\n"
    pre += "spec=" + json.dumps(spec) + "\n"
    pre += "SOURCES=" + json.dumps(sources) + "\n"
    res = nb.run([pre + code])
    return JSONResponse({"ok":True,"artifacts":res.artifacts or {}})
  except Exception as e:
    return JSONResponse({"ok":False,"error":str(e)}, status_code=400)
  finally:
    nb.stop()

@router.post("/docs/upload")
async def docs_upload(file: UploadFile = File(...)):
  try:
    b = await file.read()
    text = ""
    if file.filename.lower().endswith(".pdf"):
      reader = PdfReader(io.BytesIO(b))
      for p in reader.pages[:12]:
        text += (p.extract_text() or "") + "\n"
    else:
      try: text = b.decode("utf-8", errors="ignore")[:10000]
      except Exception: text = ""
    if not text.strip(): text="(no extractable text)"
    _INGEST["text"]=text; _INGEST["images"]=[]
    return JSONResponse({"ok":True,"chars":len(text)})
  except Exception as e:
    return JSONResponse({"ok":False,"error":str(e)}, status_code=400)

@router.post("/docs/upload_image")
async def docs_upload_image(file: UploadFile = File(...)):
  try:
    b = await file.read()
    b64 = "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    _INGEST["images"].append(b64)
    return JSONResponse({"ok":True,"images":len(_INGEST['images'])})
  except Exception as e:
    return JSONResponse({"ok":False,"error":str(e)}, status_code=400)

@router.post("/docs/summarize")
async def docs_summarize(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  if _risky(q): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)
  body = "Document excerpt:\n" + (_INGEST["text"][:4000] or "(empty)") + f"\nImages: {len(_INGEST['images'])}\n"
  srcs = await _scholar_search(q or "cyber security", per=5)
  try:
    out = gen_text(DOC_RULES + "\nCite sources inline by [#] where useful.", body, model=MODEL) or ""
    return JSONResponse({"ok":True,"summary":out.strip(),"sources":srcs})
  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)

# ===================== UI =====================

@router.get("/scholarviz", response_class=HTMLResponse)
def scholarviz(request: Request, q: str = Query("", description="Ask the model")):
  page = """
<!doctype html><html><head>
<meta charset="utf-8"/><title>ScholarViz</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
body{font-family:ui-sans-serif,system-ui,-apple-system;margin:0} header{background:#0f172a;color:#fff;padding:12px 16px}
main{max-width:960px;margin:0 auto;padding:16px} .card{border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:12px 0}
textarea,input,button,select{font-size:14px} textarea,input{border:1px solid #e5e7eb;border-radius:8px;padding:8px}
button{background:#111827;color:#fff;border:none;border-radius:8px;padding:8px 12px;cursor:pointer} button:hover{background:#0f172a}
#diagram_view svg{max-width:100%} .src a{text-decoration:none}
</style>
<script>mermaid.initialize({startOnLoad:false});</script>
</head><body>
<header><h1 style="margin:0;font-size:18px">ScholarViz</h1></header>
<main>

<div class="card">
  <h3 style="margin:0 0 8px">Ask the LLM</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <input id="ask_q" placeholder="e.g., Summarize Zero Trust in 120 words" style="flex:1;min-width:240px"/>
    <button id="ask_go">Ask</button>
  </div>
  <pre id="ask_out" style="white-space:pre-wrap;margin-top:10px"></pre>
  <div id="ask_src" class="src" style="margin-top:6px"></div>
  <button id="ask_to_nb" style="margin-top:6px;display:none">Send sources to Notebook</button>
</div>

<div class="card">
  <h3 style="margin:0 0 8px">Plan a Diagram</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <input id="plan_q" placeholder="e.g., CIA triad with explicit labels" style="flex:1;min-width:240px"/>
    <button id="plan_go">Plan</button>
    <button id="plan_grounded">Grounded Plan (from PDFs)</button>
  </div>
  <p id="plan_sum" style="color:#6b7280"></p>
  <div id="plan_src" class="src" style="margin-top:6px"></div>

  <!-- New grounded-output area -->
  <pre id="plan_g_out" style="white-space:pre-wrap;margin-top:10px"></pre>

  <div id="diagram_view" class="card" style="display:none"></div>
</div>


<div class="card">
  <h3 style="margin:0 0 8px">Ask about this Diagram</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <input id="d_q" placeholder="e.g., Where does WebAuthn help most?" style="flex:1;min-width:240px"/>
    <button id="d_go" disabled>Ask</button>
  </div>
  <pre id="d_out" style="white-space:pre-wrap;margin-top:10px"></pre>
</div>

<div class="card">
  <h3 style="margin:0 0 8px">Notebook + Papers/Diagrams</h3>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <input id="doc_file" type="file"/><button id="doc_up">Upload PDF/Text</button>
    <input id="img_file" type="file" accept="image/*"/><button id="img_up">Upload Diagram</button>
    <input id="doc_q" placeholder="Ask about the uploaded material…" style="flex:1;min-width:240px"/>
    <button id="doc_sum">Summarize/Answer</button>
  </div>
  <pre id="doc_out" style="white-space:pre-wrap;margin-top:10px"></pre>
  <div id="doc_src" class="src" style="margin-top:6px"></div>

  <div class="card" style="margin-top:10px">
    <h4 style="margin:0 0 8px">Notebook</h4>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <select id="nb_tpl"><option value="">Templates…</option></select><button id="nb_load">Load</button>
    </div>
    <textarea id="nb_code" style="width:100%;height:160px;margin-top:8px"></textarea>
    <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
      <button id="nb_run">Run</button>
      <input id="nb_spec" placeholder="(optional) spec JSON" style="flex:1;min-width:240px"/>
    </div>
    <pre id="nb_out" style="white-space:pre-wrap;margin-top:10px"></pre>
  </div>
</div>

</main>
<script>
let CURRENT_MERMAID = null;
let LAST_SOURCES = [];

async function post(path, body, form){
  const res = await fetch(path, {method:'POST', headers: form?undefined:{'Content-Type':'application/json'}, body: form?body:JSON.stringify(body||{})});
  return res.json();
}
function renderSources(el, src){
  if (!src || !src.length){ el.innerHTML=''; return; }
  el.innerHTML = '<strong>Sources:</strong><ol>' + src.map((s,i)=>(
    `<li><a href="${s.url||'#'}" target="_blank">[${i+1}] ${s.title} (${s.year||''})</a>` +
    (s.authors?` — <span style="color:#6b7280">${s.authors}</span>`:'') +
    (s.venue?` <em style="color:#6b7280">· ${s.venue}</em>`:'') + `</li>`
  )).join('') + '</ol>';
}

document.getElementById('ask_go').onclick = async ()=>{
  const q = document.getElementById('ask_q').value.trim();
  document.getElementById('ask_out').textContent='…';
  const d = await post('/ask', {q});
  if (!d.ok){ document.getElementById('ask_out').textContent='Error: '+d.error; return; }
  document.getElementById('ask_out').textContent = (d.answer||'(no answer)');
  LAST_SOURCES = d.sources||[];
  renderSources(document.getElementById('ask_src'), LAST_SOURCES);
  document.getElementById('ask_to_nb').style.display = LAST_SOURCES.length? 'inline-block':'none';
};
document.getElementById('ask_to_nb').onclick = ()=>{
  const code = "import pandas as pd\\nARTIFACTS['sources_table']=pd.DataFrame(SOURCES).to_dict(orient='records')";
  document.getElementById('nb_code').value = code;
  document.getElementById('nb_out').textContent = "Loaded a cell to show sources as a table. Click Run.";
};

document.getElementById('plan_go').onclick = async ()=>{
  const q = document.getElementById('plan_q').value.trim();
  document.getElementById('plan_sum').textContent='…';
  const d = await post('/diagram/plan', {q});
  if (!d.ok){ document.getElementById('plan_sum').textContent='Error: '+d.error; return; }
  document.getElementById('plan_sum').textContent = d.summary||'';
  renderSources(document.getElementById('plan_src'), d.sources||[]);
  const view = document.getElementById('diagram_view');
  if (d.make_diagram && d.mermaid){
    CURRENT_MERMAID = d.mermaid;
    view.style.display='block'; view.textContent='';
    mermaid.render('schviz_svg', d.mermaid).then(r=>view.innerHTML=r.svg).catch(_=>view.textContent='Mermaid error');
    document.getElementById('d_go').disabled = false;
  } else {
    CURRENT_MERMAID = null; view.style.display='none';
    document.getElementById('d_go').disabled = true;
  }
};

/* NEW: grounded plan click handler — placed immediately after plan_go */
document.getElementById('plan_grounded').onclick = async ()=>{
  const q = document.getElementById('plan_q').value.trim();
  const out = document.getElementById('plan_g_out');
  const view = document.getElementById('diagram_view');
  out.textContent = 'Fetching OA PDFs and synthesizing…';
  try{
    const d = await fetch('/diagram/grounded', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q})
    }).then(r=>r.json());

    if (!d.ok){
      out.textContent = 'Error: ' + d.error;
      CURRENT_MERMAID = null;
      document.getElementById('d_go').disabled = true;
      return;
    }

    out.textContent = (d.summary||'') + (d.cites && d.cites.length ? `\\nCites: ${JSON.stringify(d.cites)}` : '');

    if (d.mermaid){
      CURRENT_MERMAID = d.mermaid;
      view.style.display='block'; view.textContent='';
      mermaid.render('schviz_gsvg', d.mermaid)
        .then(r=>view.innerHTML=r.svg)
        .catch(_=>view.textContent='Mermaid error');
      document.getElementById('d_go').disabled=false;
    } else {
      CURRENT_MERMAID = null;
      view.style.display='none';
      document.getElementById('d_go').disabled=true;
    }

    renderSources(document.getElementById('plan_src'), d.sources||[]);

    const lis = (d.snippets||[])
      .map((s,i)=>`[${i}] ${s.text.substring(0,300)}${s.text.length>300?'…':''}`)
      .join('\\n\\n');
    out.textContent += lis ? `\\n\\nTop snippets:\\n${lis}` : '';

  } catch(e){
    out.textContent = 'Error: ' + e.message;
  }
};

document.getElementById('d_go').onclick = async ()=>{
  const q = document.getElementById('d_q').value.trim();
  const out = document.getElementById('d_out');
  if (!CURRENT_MERMAID){ out.textContent = "No diagram loaded. Plan a diagram first."; return; }
  out.textContent='…';
  const d = await post('/diagram/ask', { q, mermaid: CURRENT_MERMAID });
  out.textContent = d.ok ? (d.answer||'(no answer)') : ('Error: '+d.error);
};

document.getElementById('doc_up').onclick = async ()=>{
  const f = document.getElementById('doc_file').files[0];
  if (!f){ document.getElementById('doc_out').textContent='Choose a PDF/text file'; return; }
  const fd = new FormData(); fd.append('file', f);
  document.getElementById('doc_out').textContent='Uploading…';
  const d = await post('/docs/upload', fd, true);
  document.getElementById('doc_out').textContent = d.ok? ('Uploaded. Extracted chars: '+d.chars):('Error: '+d.error);
};

document.getElementById('img_up').onclick = async ()=>{
  const f = document.getElementById('img_file').files[0];
  if (!f){ document.getElementById('doc_out').textContent='Choose an image'; return; }
  const fd = new FormData(); fd.append('file', f);
  document.getElementById('doc_out').textContent='Uploading image…';
  const d = await post('/docs/upload_image', fd, true);
  document.getElementById('doc_out').textContent = d.ok? ('Images stored: '+d.images):('Error: '+d.error);
};

document.getElementById('doc_sum').onclick = async ()=>{
  const q = document.getElementById('doc_q').value.trim();
  document.getElementById('doc_out').textContent='Summarizing…';
  const d = await fetch('/docs/summarize', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({q})}).then(r=>r.json());
  document.getElementById('doc_out').textContent = d.ok? (d.summary||'(no summary)'):('Error: '+d.error);
  renderSources(document.getElementById('doc_src'), d.sources||[]);
};

(async function nbInit(){
  const sel=document.getElementById('nb_tpl'), load=document.getElementById('nb_load'), run=document.getElementById('nb_run'), out=document.getElementById('nb_out');
  if (!sel||!load||!run) return;
  try{
    const d=await fetch('/notebook/templates').then(r=>r.json());
    if (d.ok){ d.templates.forEach(t=>{ const o=document.createElement('option'); o.value=t.id; o.textContent=t.label; o.dataset.code=t.code; sel.appendChild(o); }); }
  }catch(_){}
  load.onclick=()=>{ const o=sel.options[sel.selectedIndex]; const c=o&&o.dataset.code; if(c) document.getElementById('nb_code').value=c; };
  run.onclick=async ()=>{
    const code=document.getElementById('nb_code').value.trim();
    const specRaw=document.getElementById('nb_spec').value.trim();
    let spec={}; if (specRaw){ try{ spec=JSON.parse(specRaw);}catch(_){ }}
    out.textContent='Running…';
    const d=await post('/notebook/run', {code, spec, sources: LAST_SOURCES });
    out.textContent = d.ok? JSON.stringify(d.artifacts||{}, null, 2) : ('Error: '+d.error);
  };
})();
</script>
</body></html>
"""
  return HTMLResponse(page)

@router.get("/studio", response_class=HTMLResponse)
def studio_alias(request: Request, q: str = Query("", description="Ask the model")):
    return scholarviz(request, q)

@router.get("/", response_class=HTMLResponse)
def root_alias(request: Request):
    # optional: land on ScholarViz
    return scholarviz(request, q="")

# ---------- OA PDF helpers ----------
async def _open_access_hits(q: str, per: int = 6) -> List[dict]:
  url = "https://api.openalex.org/works"
  now = datetime.date.today()
  start = f"{now.year-4}-01-01"
  params = {
    "search": q,
    "filter": f"from_publication_date:{start},language:en,primary_location.source.type:journal",
    "sort": "publication_date:desc",
    "per_page": per
  }
  try:
    async with httpx.AsyncClient(timeout=14.0) as hx:
      r = await hx.get(url, params=params); r.raise_for_status()
      res = r.json().get("results", [])
  except Exception:
    return []
  out = []
  for w in res:
    loc = w.get("primary_location",{}) or {}
    oa = w.get("open_access",{}) or {}
    url_pdf = (loc.get("pdf_url") or oa.get("oa_url") or "")
    out.append({
      "title": w.get("title") or "(untitled)",
      "year": w.get("publication_year") or "",
      "venue": (loc.get("source",{}) or {}).get("display_name"),
      "url": loc.get("landing_page_url") or oa.get("oa_url") or w.get("id"),
      "pdf": url_pdf if (url_pdf and url_pdf.lower().endswith(".pdf")) else "",
      "authors": ", ".join(a.get("author",{}).get("display_name","") for a in (w.get("authorships") or [])[:4])
    })
  return [h for h in out if h.get("pdf")]

def _split_blocks(txt: str) -> List[str]:
  txt = re.sub(r"\s+", " ", txt or "").strip()
  blocks = re.split(r"(?<=\.)\s{2,}|(?:\n\s*){2,}", txt)
  return [b.strip() for b in blocks if len(b.strip()) > 120][:120]

def _fig_captions(raw: str) -> List[str]:
  caps = re.findall(r"(?:Figure|Fig\.?)\s*\d+[:\.\)]\s*[^\n\.]+(?:\.[^\n\.]+)?", raw, flags=re.I)
  return [c.strip() for c in caps][:40]

def _rank_snippets(q: str, snippets: List[str], k: int = 10) -> List[dict]:
  try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vec = TfidfVectorizer(stop_words="english", max_features=8000)
    X = vec.fit_transform(snippets + [q])
    sims = (X[:-1] @ X[-1].T).toarray().ravel()
    idx = np.argsort(-sims)[:k]
    return [{"i": int(i), "text": snippets[int(i)], "score": float(sims[int(i)])} for i in idx]
  except Exception:
    ql = set(re.findall(r"[a-z0-9]+", q.lower()))
    scored = []
    for i, s in enumerate(snippets):
      sl = set(re.findall(r"[a-z0-9]+", s.lower()))
      scored.append((i, len(ql & sl)))
    scored.sort(key=lambda t: -t[1])
    return [{"i": i, "text": snippets[i], "score": float(sc)} for i, sc in scored[:k]]

async def _fetch_pdf_text(url: str) -> str:
  try:
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
      r = await hx.get(url); r.raise_for_status()
      b = r.content
  except Exception:
    return ""
  txt = ""
  try:
    if PdfReader is not None:
      reader = PdfReader(io.BytesIO(b))
      for p in reader.pages[:20]: txt += (p.extract_text() or "") + "\n"
    elif 'fitz' in globals() and fitz is not None:
      doc = fitz.open(stream=b, filetype="pdf")
      for i, page in enumerate(doc):
        if i >= 20: break
        txt += (page.get_text() or "") + "\n"
  except Exception:
    pass
  return txt

GROUNDED_RULES = (
  "Use provided snippets to synthesize a small Mermaid diagram (5–8 nodes, LR) with 2–4 dashed control nodes."
  " Output JSON: {mermaid:string, summary:string, cites:number[]}."
  " Each node/edge choice must be inferable from snippets; cite with indices in 'cites' and reference them inline as [#] in the summary."
  " Do not invent procedures; defense-only. If snippets are insufficient, say so and set mermaid to ''."
)

@router.post("/diagram/grounded")
async def diagram_grounded(payload:dict=Body(...)):
  q = (payload or {}).get("q") or ""
  if _risky(q): return JSONResponse({"ok":False,"error":"Blocked for safety."}, status_code=400)

  hits = await _open_access_hits(q, per=8)
  if not hits: return JSONResponse({"ok":False,"error":"No OA PDFs found"}, status_code=404)

  texts = []
  for h in hits[:3]:
    t = await _fetch_pdf_text(h["pdf"])
    if t: texts.append((h, t))

  if not texts: return JSONResponse({"ok":False,"error":"Could not extract text from OA PDFs"}, status_code=502)

  pool = []
  for h, t in texts:
    caps = _fig_captions(t)
    blks = _split_blocks(t)
    pool.extend([f"[cap] {c}" for c in caps] + blks)

  top = _rank_snippets(q, pool, k=10)
  bundle = {"query": q, "snippets": [{"idx": s["i"], "text": s["text"]} for s in top]}

  try:
    user = json.dumps(bundle, ensure_ascii=False)
    res = gen_json(GROUNDED_RULES, user, model=MODEL) or {}
    mer = _sanitize_mermaid((res or {}).get("mermaid") or "")
    summ = (res or {}).get("summary") or ""
    cites = (res or {}).get("cites") or []
    return JSONResponse({
      "ok": True,
      "mermaid": mer,
      "summary": summ,
      "cites": cites,
      "snippets": bundle["snippets"],
      "sources": hits  # show which PDFs were used
    })
  except Exception as e:
    return JSONResponse({"ok":False,"error":f"llm: {e}"}, status_code=500)
