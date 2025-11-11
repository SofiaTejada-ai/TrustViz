// trustviz/static/lesson_player.js
// Lightweight zyBooks-style player over Cytoscape.
// Supports two modes:
//   1) topic → /lesson/plan (LLM-only)
//   2) grounded mermaid → /lesson/plan_from_mermaid (deterministic)

const LESSON_THEME = {
  dim: 'rgba(37, 99, 235, 0.14)',    // soft blue wash instead of black
  focusStroke: '#2563eb',            // blue-600 accents
  panelBg: 'rgba(37, 99, 235, 0.06)',
  panelBorder: '#2563eb'
};

let cy, LESSON = null, idx = 0;
let DIM_EL = null; // optional dim overlay

function ensureDimLayer() {
  if (DIM_EL) return DIM_EL;
  const host = document.getElementById('diagram_cy');
  if (!host) return null;
  // Create an absolutely-positioned overlay inside the same container.
  // We default to hidden so it never blocks unless a lesson/frame asks for it.
  const dim = document.createElement('div');
  dim.id = 'lessonDim';
  Object.assign(dim.style, {
    position: 'absolute',
    inset: '0',
    borderRadius: '12px',
    background: LESSON_THEME.dim,
    display: 'none',
    pointerEvents: 'none'
  });
  // Ensure host can position children
  const parentStyle = getComputedStyle(host);
  if (parentStyle.position === 'static') host.style.position = 'relative';
  host.appendChild(dim);
  DIM_EL = dim;
  return DIM_EL;
}

function showDim(on) {
  if (!DIM_EL) return;
  DIM_EL.style.display = on ? 'block' : 'none';
}

async function fetchLessonFromTopic(topic){
  const res = await fetch("/lesson/plan", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ topic })
  });
  return await res.json();
}

async function fetchLessonFromMermaid(mermaid){
  const res = await fetch("/lesson/plan_from_mermaid", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ mermaid })
  });
  return await res.json();
}

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
          'background-opacity': 0.2,
          'border-width': 1,
          'border-color': '#334155',
          'color': '#111827',
          'transition-property': 'background-color,opacity',
          'transition-duration': 300
      }},
      { selector: 'edge', style: {
          'curve-style':'bezier',
          'target-arrow-shape':'triangle',
          'opacity': 0.7,
          'width': 1.2,
          'line-color': '#475569',
          'target-arrow-color': '#475569',
          'transition-property': 'line-color,opacity',
          'transition-duration': 300
      }},
      // Focus (blue, not black)
      { selector: '.hl, .focus', style: {
          'background-color': LESSON_THEME.focusStroke,
          'border-color': LESSON_THEME.focusStroke,
          'line-color': LESSON_THEME.focusStroke,
          'target-arrow-color': LESSON_THEME.focusStroke,
          'background-opacity': 1,
          'opacity': 1
      }},
      { selector: '.fade', style: { 'opacity': 0.15 } }
    ],
    layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.2 }
  });

  return cy;
}

function applyAdds(adds){
  if (!adds) return;
  const toAdd = [];
  if (Array.isArray(adds.nodes)){
    for (const n of adds.nodes){
      toAdd.push({ group:'nodes', data: n });
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

function renderFrame(i){
  if (!LESSON || !LESSON.frames || !LESSON.frames[i]) return;
  const f = LESSON.frames[i];
  const title = document.getElementById("lessonTitle");
  const note  = document.getElementById("lessonNote");
  const quiz  = document.getElementById("lessonQuiz");

  title.textContent = `${i+1}/${LESSON.frames.length} — ${f.title || ''}`;
  note.textContent  = f.note || '';

  applyAdds(f.adds);
  applyHighlight(f.highlight);
  applyFade(f.fade);

  // Optional blue dimmer for spotlight-style frames
  // Enable by setting f.dim === true in a frame (default off).
  showDim(Boolean(f.dim));

  if (f.quiz && Array.isArray(f.quiz.choices)) {
    quiz.style.display = "block";
    quiz.innerHTML = `
      <div style="font-weight:600">${f.quiz.prompt || ""}</div>
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

export async function initLessonPlayer(opts = {}){
  // Hide Mermaid SVG; show the Cytoscape canvas
  const merDiv = document.getElementById("diagram_view");
  const cyDiv  = document.getElementById("diagram_cy");
  if (merDiv) merDiv.style.display = "none";
  if (cyDiv)  cyDiv.style.display  = "block";

  initCy("diagram_cy");
  ensureDimLayer(); // creates (hidden) blue dim overlay

  // Determine source
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

  // Wire buttons (idempotent)
  const prev = document.getElementById("lessonPrev");
  const next = document.getElementById("lessonNext");
  if (prev) prev.onclick = ()=>{ if (idx>0){ idx--; renderFrame(idx); } };
  if (next) next.onclick = ()=>{ if (LESSON.frames && idx<LESSON.frames.length-1){ idx++; renderFrame(idx); } };

  // Seed initial graph if present
  if (LESSON.initial) {
    applyAdds(LESSON.initial);
  }

  // Global lesson-level dim default (optional)
  showDim(Boolean(LESSON.dim));

  renderFrame(idx);
}
