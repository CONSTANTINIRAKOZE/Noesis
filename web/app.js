// Walter's frontend. Walter himself runs right here in the browser (walter.js),
// so this page works as a static site with no server.
import { Walter } from './walter.js';

const $ = (id) => document.getElementById(id);
const show = (ch) => (ch === '.' ? 'end' : ch);

let walter;
const ready = Walter.load('.').then((w) => { walter = w; });
// Give the browser a moment to paint "thinking…" before the maths runs.
const tick = () => new Promise((r) => setTimeout(r, 0));

function guard(errorEl, fn) {
  return async (...args) => {
    errorEl.textContent = '';
    try { await ready; await tick(); await fn(...args); } catch (e) { errorEl.textContent = e.message; }
  };
}

function bars(el, dist, { onPick, limit } = {}) {
  el.innerHTML = '';
  const max = Math.max(...dist.map((d) => d.p));
  for (const d of dist.slice(0, limit || dist.length)) {
    const row = document.createElement('div');
    row.className = 'bar';
    row.title = onPick ? (d.char === '.' ? 'end the name here' : `add "${d.char}"`) : '';
    row.innerHTML = `<span class="ch">${show(d.char)}</span>
      <span class="track"><span class="fill" style="width:${(100 * d.p) / max}%"></span></span>
      <span class="val">${(100 * d.p).toFixed(1)}%</span>`;
    if (onPick) row.addEventListener('click', () => onPick(d.char));
    el.appendChild(row);
  }
}

// ---- tabs -----------------------------------------------------------------
const loaded = {};
document.querySelectorAll('nav button').forEach((btn) => btn.addEventListener('click', () => {
  document.querySelectorAll('nav button').forEach((b) => b.setAttribute('aria-selected', b === btn));
  document.querySelectorAll('.panel').forEach((p) => { p.hidden = p.id !== `tab-${btn.dataset.tab}`; });
  if (!loaded[btn.dataset.tab]) { loaded[btn.dataset.tab] = true; onFirstOpen[btn.dataset.tab]?.(); }
}));

// ---- header ---------------------------------------------------------------
ready.then(() => walter.info()).then((i) => {
  $('stats').textContent = `${i.n_layer} layers · ${i.n_head} heads each · ${i.params.toLocaleString()} parameters · `
    + `loss ${i.dev_loss} on unseen names · trained on ${i.n_train_names.toLocaleString()} names`;
}).catch(() => { $('stats').textContent = 'could not load Walter'; });

// ---- Invent ---------------------------------------------------------------
$('gen-temp').addEventListener('input', (e) => { $('gen-temp-val').textContent = Number(e.target.value).toFixed(1); });
const invent = guard($('gen-error'), async () => {
  const n = Math.min(50, Math.max(1, Number($('gen-n').value) || 1));
  const names = walter.generate(n, Number($('gen-temp').value), $('gen-prefix').value);
  $('gen-out').innerHTML = names.map((n) => `<li><span>${n.name || '(empty)'}</span>${
    n.known ? '<span class="pill known">real name</span>' : '<span class="pill new">new</span>'}</li>`).join('');
});
$('gen-form').addEventListener('submit', (e) => { e.preventDefault(); invent(); });

// ---- Predict --------------------------------------------------------------
const predict = guard($('pred-error'), async () => {
  const prefix = $('pred-prefix').value.toLowerCase();
  const out = walter.nextLetter(prefix);
  bars($('pred-chart'), out.distribution, {
    onPick: (ch) => {
      if (ch === '.') { $('pred-error').textContent = `Walter would end the name here: "${prefix}".`; return; }
      $('pred-prefix').value = prefix + ch;
      predict();
    },
  });
});
$('pred-prefix').addEventListener('input', predict);
$('pred-clear').addEventListener('click', () => { $('pred-prefix').value = ''; predict(); });

// ---- Score ----------------------------------------------------------------
const score = guard($('score-error'), async () => {
  const s = walter.score($('score-name').value);
  const maxS = Math.max(...s.steps.flatMap((t) => [t.surprise_gpt, t.surprise_bigram]), 4);
  const where = s.in_training ? 'in Walter\'s training names' : s.in_dataset ? 'in the dataset, but not in training' : 'not in the dataset';
  $('score-out').innerHTML = `
    <div class="summary">
      <div class="card"><div class="num" style="color:var(--gpt)">${s.loss_gpt.toFixed(2)}</div><div class="lbl">GPT surprise per letter</div></div>
      <div class="card"><div class="num" style="color:var(--bigram)">${s.loss_bigram.toFixed(2)}</div><div class="lbl">bigram (Rung 1) surprise</div></div>
      <div class="card"><div class="num">${where.startsWith('not') ? 'unseen' : 'seen'}</div><div class="lbl">${where}</div></div>
    </div>
    <div class="legend"><span><i style="background:var(--gpt)"></i>GPT</span><span><i style="background:var(--bigram)"></i>bigram</span><span>bar height = surprise · below: Walter's top guess</span></div>
    <div class="letters">${s.steps.map((t) => `
      <div class="letter" title="after '${t.context}': GPT gave '${show(t.char)}' ${(100 * t.p_gpt).toFixed(1)}%, bigram ${(100 * t.p_bigram).toFixed(1)}%">
        <div class="cols">
          <div class="col gpt" style="height:${(100 * t.surprise_gpt) / maxS}%"></div>
          <div class="col bigram" style="height:${(100 * t.surprise_bigram) / maxS}%"></div>
        </div>
        <div class="ch">${show(t.char)}</div>
        <div class="guess">${(100 * t.p_gpt).toFixed(0)}%<br>guess: ${show(t.top[0].char)}</div>
      </div>`).join('')}
    </div>`;
});
$('score-form').addEventListener('submit', (e) => { e.preventDefault(); score(); });

// ---- Inside Walter --------------------------------------------------------
let ablated = new Set();   // "layer,head"
const inspect = guard($('in-error'), async () => {
  const ablate = [...ablated].map((k) => k.split(',').map(Number));
  const r = walter.inspect($('in-name').value, ablate);
  const T = r.tokens.length;
  const heads = $('in-heads');
  heads.innerHTML = '';
  r.attention.forEach((layer, l) => layer.forEach((att, h) => {
    const key = `${l},${h}`;
    const div = document.createElement('div');
    div.className = `head${ablated.has(key) ? ' off' : ''}`;
    div.title = ablated.has(key) ? 'click to switch this head back on' : 'click to switch this head off';
    let cells = '<span></span>' + r.tokens.map((c) => `<span class="lab">${c}</span>`).join('');
    att.forEach((row, i) => {
      cells += `<span class="lab">${r.tokens[i]}</span>`;
      cells += row.map((v, j) => `<span class="cell" style="background:${j > i ? 'transparent' : `rgba(var(--heat),${v.toFixed(3)})`}"></span>`).join('');
    });
    div.innerHTML = `<div class="title">L${l} H${h}${ablated.has(key) ? ' · off' : ''}</div>
      <div class="grid" style="grid-template-columns:repeat(${T + 1},1fr)">${cells}</div>`;
    div.addEventListener('click', () => { ablated.has(key) ? ablated.delete(key) : ablated.add(key); inspect(); });
    heads.appendChild(div);
  }));
  bars($('in-next'), r.next);
  $('in-lens').innerHTML = r.lens.map((s) => `<tr><td>${s.stage}</td><td>${
    s.top.slice(0, 3).map((g) => `<span class="g">${show(g.char)} ${(100 * g.p).toFixed(0)}%</span>`).join('')}</td></tr>`).join('');
});
$('in-form').addEventListener('submit', (e) => { e.preventDefault(); inspect(); });
$('in-reset').addEventListener('click', () => { ablated = new Set(); inspect(); });

// ---- first load -----------------------------------------------------------
const onFirstOpen = { predict, score, inside: inspect };
loaded.invent = true;
invent();
