// trustviz/static/lesson_player.js
// zyBooks-style lesson player over Cytoscape with group-based node shapes/colors.

const LESSON_THEME = {
  dim: 'rgba(37, 99, 235, 0.14)',
  focusStroke: '#2563eb',
  panelBg: 'rgba(37, 99, 235, 0.06)',
  panelBorder: '#2563eb'
};

// ---- Group → shape/color mapping ------------------------------------------
const GROUP_STYLE = {
  policy:  { shape: 'hexagon',          color: '#2563EB' }, // blue
  user:    { shape: 'triangle',         color: '#F97316' }, // orange
  data:    { shape: 'rectangle',        color: '#22C55E' }, // green
  monitor: { shape: 'diamond',          color: '#A78BFA' }, // purple
  service: { shape: 'round-rectangle',  color: '#EAB308' }, // yellow
  control: { shape: 'vee',              color: '#06B6D4' }, // cyan
};
const DEFAULT_NODE_STYLE = { shape: 'round-rectangle', color: '#94A3B8' }; // slate

function applyGroupStyle(n){
  const g = (n.group||n.cohort||n.role||'').toString().toLowerCase();
  const s = GROUP_STYLE[g] || DEFAULT_NODE_STYLE;
  if (!n.shape) n.shape = s.shape;
  if (!n.color) n.color = s.color;
  return n;
}

// ---- Callout palette for notes --------------------------------------------
const NOTE_COLORS = {
  info:   {bg: "#DBEAFE", border:"#2563EB", text:"#0F172A"},
  success:{bg: "#DCFCE7", border:"#16A34A", text:"#052E16"},
  warn:   {bg: "#FEF3C7", border:"#D97706", text:"#111827"},
  danger: {bg: "#FEE2E2", border:"#DC2626", text:"#111827"},
  purple: {bg: "#EDE9FE", border:"#7C3AED", text:"#1F1537"}
};
const NOTE_STYLES = Object.keys(NOTE_COLORS);
function _autoNoteStyle(i){ return NOTE_STYLES[i % NOTE_STYLES.length]; }

function calloutHTML({note="", style="info", shape="card", icon=""}){
  const c = NOTE_COLORS[style] || NOTE_COLORS.info;
  const base = `background:${c.bg};border:2px solid ${c.border};color:${c.text};padding:10px 12px;line-height:1.35;font-size:14px;`;
  let box = "", extra = "";
  if (shape === "pill") box = `border-radius:9999px;`;
  else if (shape === "ribbon") box = `border-radius:10px;border-left-width:8px;`;
  else if (shape === "sticky"){ box = `border-radius:6px;transform:rotate(-0.6deg);box-shadow:0 4px 10px rgba(0,0,0,.08);`; }
  else if (shape === "bubble"){ box = `border-radius:14px;position:relative;`; extra = `<svg width="18" height="12" viewBox="0 0 18 12" style="position:absolute;left:18px;bottom:-10px"><path d="M0,0 L18,0 L10,12 Z" fill="${c.bg}" stroke="${c.border}" stroke-width="2"></path></svg>`; }
  else box = `border-radius:12px;`;
  const iconSpan = icon ? `<span style="font-weight:700;margin-right:8px">${icon}</span>` : "";
  return `<div style="${base}${box}">${iconSpan}${note}</div>${extra}`;
}

let cy, LESSON = null, idx = 0;
let DIM_EL = null;

function ensureDimLayer() {
  if (DIM_EL) return DIM_EL;
  const host = document.getElementById('diagram_cy');
  if (!host) return null;
  const dim = document.createElement('div');
  dim.id = 'lessonDim';
  Object.assign(dim.style, { position:'absolute', inset:'0', borderRadius:'12px', background:LESSON_THEME.dim, display:'none', pointerEvents:'none' });
  if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
  host.appendChild(dim);
  DIM_EL = dim;
  return DIM_EL;
}
function showDim(on){ if (DIM_EL) DIM_EL.style.display = on ? 'block' : 'none'; }

async function fetchLessonFromTopic(topic){
  const res = await fetch("/lesson/plan",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic})});
  return await res.json();
}
async function fetchLessonFromMermaid(mermaid){
  const res = await fetch("/lesson/plan_from_mermaid",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mermaid})});
  return await res.json();
}

// ---- Cytoscape init with data-mapped shape/color --------------------------
function initCy(containerId="diagram_cy"){
  if (cy) return cy;
  const el = document.getElementById(containerId);
  if (!el) throw new Error(`#${containerId} not found`);

  cy = cytoscape({
    container: el,
    style: [
      { selector: 'node', style: {
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'shape': 'data(shape)',
          'background-color': 'data(color)',
          'background-opacity': 0.9,
          'border-width': 2,
          'border-color': '#1F2937',
          'color': '#0B1020',
          'font-size': 12,
          'padding': 8,
          'transition-property': 'background-color,opacity,border-color',
          'transition-duration': 300
      }},
      { selector: 'edge', style: {
          'curve-style':'bezier',
          'target-arrow-shape':'triangle',
          'opacity': 0.75,
          'width': 1.4,
          'line-color': '#475569',
          'target-arrow-color': '#475569',
          'transition-property': 'line-color,opacity',
          'transition-duration': 300
      }},
      { selector: '.hl, .focus', style: {
          'border-color': LESSON_THEME.focusStroke,
          'line-color': LESSON_THEME.focusStroke,
          'target-arrow-color': LESSON_THEME.focusStroke,
          'opacity': 1
      }},
      { selector: '.fade', style: { 'opacity': 0.18 } }
    ],
    layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.2 }
  });

  return cy;
}

function applyAdds(adds){
  if (!adds) return;
  const toAdd = [];
  if (Array.isArray(adds.nodes)){
    for (const n0 of adds.nodes){
      const n = {...n0};                // copy so we can mutate safely
      if (n.data) {                     // support {data:{id,label,group}}
        n.data = applyGroupStyle(n.data);
        toAdd.push({ group:'nodes', data: n.data });
      } else {
        applyGroupStyle(n);
        toAdd.push({ group:'nodes', data: n });
      }
    }
  }
  if (Array.isArray(adds.edges)){
    for (const e of adds.edges){
      toAdd.push({ group:'edges', data: e });
    }
  }
  if (toAdd.length) {
    cy.add(toAdd);
    cy.layout({ name:'breadthfirst', directed:true, spacingFactor: 1.2 }).run();
  }
}

function applyHighlight(h){
  cy.elements().removeClass('hl').removeClass('focus');
  if (!h) return;
  (h.nodes||[]).forEach(id => cy.$id(id).addClass('hl'));
  (h.edges||[]).forEach(id => cy.$id(id).addClass('hl'));
}
function applyFade(f){
  cy.elements().removeClass('fade');
  if (!f) return;
  (f.nodes||[]).forEach(id => cy.$id(id).addClass('fade'));
  (f.edges||[]).forEach(id => cy.$id(id).addClass('fade'));
}

// ---- Frame renderer (styled notes + quiz) ---------------------------------
function renderFrame(i){
  if (!LESSON || !LESSON.frames || !LESSON.frames[i]) return;
  const f = LESSON.frames[i];
  const title = document.getElementById("lessonTitle");
  const note  = document.getElementById("lessonNote");
  const quiz  = document.getElementById("lessonQuiz");

  title.textContent = `${i+1}/${LESSON.frames.length} — ${f.title || ''}`;

  applyAdds(f.adds);
  applyHighlight(f.highlight);
  applyFade(f.fade);
  showDim(Boolean(f.dim));

  const style = f.noteStyle || _autoNoteStyle(i);
  const shape = f.noteShape || "card";
  const icon  = f.icon || "";
  note.innerHTML = calloutHTML({ note: (f.note || ""), style, shape, icon });
  try{ note.animate([{opacity:0,transform:"translateY(6px)"},{opacity:1,transform:"translateY(0)"}],{duration:220,easing:"ease-out"});}catch(_){}

  if (f.quiz && Array.isArray(f.quiz.choices)) {
    quiz.style.display = "block";
    quiz.innerHTML = `
      <div style="font-weight:600;margin-bottom:6px">${f.quiz.prompt || ""}</div>
      ${f.quiz.choices.map((c,ci)=>`<button data-ci="${ci}" class="quizChoice" style="margin:4px">${c}</button>`).join("")}
      <div id="quizMsg" style="margin-top:6px"></div>
    `;
    quiz.querySelectorAll(".quizChoice").forEach(btn=>{
      btn.onclick = ()=>{
        const m = document.getElementById("quizMsg");
        const ok = Number(btn.dataset.ci) === Number((f.quiz.answerIndex||0));
        m.textContent = ok ? "✅ Correct" : "❌ Try again";
      };
    });
  } else {
    quiz.style.display = "none";
    quiz.innerHTML = "";
  }
}

// ---- Entrypoint -----------------------------------------------------------
export async function initLessonPlayer(opts = {}){
  const merDiv = document.getElementById("diagram_view");
  const cyDiv  = document.getElementById("diagram_cy");
  if (merDiv) merDiv.style.display = "none";
  if (cyDiv)  cyDiv.style.display  = "block";

  initCy("diagram_cy");
  ensureDimLayer();

  let spec = null;
  if (opts.useMermaid && window.CURRENT_MERMAID) {
    spec = await fetchLessonFromMermaid(window.CURRENT_MERMAID);
  } else {
    const topic = (document.getElementById("plan_q")?.value || "").trim() ||
                  "Incident Response lifecycle with Playbooks as control";
    spec = await fetchLessonFromTopic(topic);
  }

  LESSON = spec || {};
  idx = 0;

  const prev = document.getElementById("lessonPrev");
  const next = document.getElementById("lessonNext");
  if (prev) prev.onclick = ()=>{ if (idx>0){ idx--; renderFrame(idx); } };
  if (next) next.onclick = ()=>{ if (LESSON.frames && idx<LESSON.frames.length-1){ idx++; renderFrame(idx); } };

  if (LESSON.initial) applyAdds(LESSON.initial);
  showDim(Boolean(LESSON.dim));
  renderFrame(idx);
}
