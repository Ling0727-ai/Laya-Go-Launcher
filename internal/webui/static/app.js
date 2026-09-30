'use strict';
// Laya Server console. Talks only to the same-origin /api/v1 admin surface.
const $ = (id) => document.getElementById(id);
const STEP_LABEL = { tensorrt: 'TensorRT', 'onnx-cuda': 'ONNX CUDA', 'onnx-directml': 'ONNX DirectML', 'onnx-cpu': 'ONNX CPU' };
const ALL_STEPS = Object.keys(STEP_LABEL);
let chain = [];
let defaults = null;

$('token').value = sessionStorage.getItem('laya-token') || '';
$('token').addEventListener('change', () => sessionStorage.setItem('laya-token', $('token').value));

async function api(method, path, body) {
  const headers = { 'Content-Type': 'application/json' };
  const tok = $('token').value.trim();
  if (tok) headers.Authorization = 'Bearer ' + tok;
  const res = await fetch('/api/v1' + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data.error && data.error.message) || res.status + ' ' + res.statusText);
  return data;
}

function toast(msg, bad) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast show' + (bad ? ' bad' : '');
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (t.className = 'toast'), 4000);
}

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

async function busy(btn, fn) {
  btn.disabled = true;
  try { await fn(); } catch (e) { toast(e.message, true); } finally { btn.disabled = false; }
}

// ── status ──────────────────────────────────────────────────────────────────
async function refreshStatus() {
  let h;
  try { h = await api('GET', '/health'); } catch (e) { $('dot').className = 'dot bad'; return; }
  const load = h.load || {};
  $('ver').textContent = 'v' + h.version;
  $('dot').className = 'dot ' + (load.phase === 'ready' ? (load.step === 'tensorrt' ? 'ok' : 'warn') : load.phase === 'loading' ? 'warn' : 'bad');
  const chips = $('chips');
  chips.replaceChildren(
    chip('GPU', h.device.name || '无'),
    chip('显存', h.device.vram_total_mb ? `${h.device.vram_free_mb}/${h.device.vram_total_mb} MiB` : '—'),
    chip('TensorRT', h.kernel.available ? h.kernel.tensorrt : '不可用'),
    chip('状态', { ready: '就绪', loading: '加载中', failed: '失败', idle: '空闲' }[load.phase] || '—'),
  );
  const run = $('run');
  run.replaceChildren();
  const rows = [
    ['步骤', STEP_LABEL[load.step] || '—'], ['模型', h.engine.loaded ? h.engine.path : '未加载'],
    ['后端', [h.engine.backend, load.device].filter(Boolean).join(' · ') || '—'],
    ['校准', load.model_config || '—'], ['说明', load.note || load.error || '—'],
  ];
  for (const [k, v] of rows) run.append(el('dt', {}, k), el('dd', {}, v));
  const ol = $('attempts');
  ol.replaceChildren(...(load.attempts || []).map((a) => el('li', { class: a.ok ? 'ok' : a.skipped ? 'skip' : 'bad' },
    `${STEP_LABEL[a.step] || a.step}: ${a.path ? a.path.split(/[\\/]/).pop() : '(无文件)'} — ${a.ok ? `成功 ${Math.round(a.ms)} ms` : a.skipped ? '跳过：' + a.error : '失败：' + a.error}`)));
  if (!ol.children.length) ol.append(el('li', { class: 'skip' }, '尚未加载'));
  return h;
}
function chip(k, v) { return el('span', { class: 'chip' }, k + ' ', el('b', {}, String(v))); }
// ── config ──────────────────────────────────────────────────────────────────
function renderChain() {
  const ul = $('chain');
  const order = [...chain, ...ALL_STEPS.filter((s) => !chain.includes(s))];
  ul.replaceChildren(...order.map((s) => {
    const on = chain.includes(s);
    const cb = el('input', { type: 'checkbox', 'aria-label': STEP_LABEL[s] });
    cb.checked = on;
    cb.addEventListener('change', () => { chain = cb.checked ? [...chain, s] : chain.filter((x) => x !== s); renderChain(); });
    const move = (d) => () => { const i = chain.indexOf(s), j = i + d; if (i < 0 || j < 0 || j >= chain.length) return; [chain[i], chain[j]] = [chain[j], chain[i]]; renderChain(); };
    return el('li', {}, cb, el('span', {}, STEP_LABEL[s]),
      el('button', { type: 'button', 'aria-label': '上移', onclick: move(-1) }, '↑'),
      el('button', { type: 'button', 'aria-label': '下移', onclick: move(1) }, '↓'));
  }));
}

function fillForm(c) {
  const f = $('cfg');
  for (const k of ['model_seq', 'backend', 'provider', 'engine_path', 'precision', 'contexts', 'engine_dir', 'http_addr']) f.elements[k].value = c[k] ?? '';
  f.elements.auto_convert.checked = !!c.auto_convert;
  f.elements.model_dirs.value = (c.model_dirs || []).join('\n');
  chain = [...(c.fallback || [])];
  renderChain();
}

async function loadConfig() {
  const r = await api('GET', '/config');
  defaults = r.defaults;
  fillForm(r.config);
  $('cfg-path').textContent = '配置文件：' + r.path + ' · 分词器：' + (r.tokenizer || '—');
}

function readForm() {
  const f = $('cfg').elements;
  if (!chain.length) throw new Error('回退链至少保留一步');
  return {
    model_seq: Number(f.model_seq.value) || 8192, backend: f.backend.value, provider: f.provider.value,
    engine_path: f.engine_path.value.trim(), precision: f.precision.value, auto_convert: f.auto_convert.checked,
    contexts: Number(f.contexts.value) || 0, engine_dir: f.engine_dir.value.trim(), http_addr: f.http_addr.value.trim(),
    model_dirs: f.model_dirs.value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean), fallback: chain,
  };
}

async function saveConfig(reload) {
  const r = await api('PUT', '/config', { ...readForm(), reload });
  toast(reload ? '已保存，正在重新加载…' : '已保存' + (r.restart_required ? '（' + r.restart_required + '）' : ''));
  fillForm(r.config);
  if (reload) poll(20);
}

$('cfg').addEventListener('submit', (e) => { e.preventDefault(); busy(e.submitter || $('cfg'), () => saveConfig(true)); });
$('btn-save').addEventListener('click', (e) => busy(e.target, () => saveConfig(false)));
$('btn-defaults').addEventListener('click', () => { if (defaults) fillForm({ ...defaults, http_addr: $('cfg').elements.http_addr.value }); });

// ── models / conversion ─────────────────────────────────────────────────────
async function refreshModels(rescan) {
  const [m, h] = await Promise.all([api(rescan ? 'POST' : 'GET', rescan ? '/models/rescan' : '/models'), api('GET', '/health')]);
  const active = h.engine.loaded ? h.engine.path.toLowerCase() : '';
  $('models').replaceChildren(...m.models.map((x) => {
    const tags = [x.multi_profile && '双 profile', x.optimized && '融合'].filter(Boolean).join(' · ');
    const btn = el('button', { type: 'button' }, '加载');
    btn.addEventListener('click', () => busy(btn, async () => {
      await api('POST', '/engine/load', { path: x.path, backend: x.format === 'engine' ? 'tensorrt' : 'onnx', provider: x.format === 'onnx' ? 'cuda' : '' });
      toast('已加载 ' + x.name); await refreshAll();
    }));
    return el('tr', { class: x.path.toLowerCase() === active ? 'active' : '' },
      el('td', { class: 'name', title: x.path }, x.name, tags ? el('div', { class: 'muted' }, tags) : null),
      el('td', {}, x.format), el('td', {}, x.seq_max || '?'), el('td', {}, x.batch_max || '—'),
      el('td', {}, x.precision || '—'), el('td', {}, Math.round(x.size_mb) + ' MB'), el('td', {}, btn));
  }));
  $('dirs').textContent = '扫描目录：' + m.dirs.join('；');
}

async function refreshConvert() {
  const { jobs } = await api('GET', '/convert');
  const j = jobs[jobs.length - 1];
  const log = $('conv-log');
  if (!j) { $('conv-state').textContent = ''; log.hidden = true; return false; }
  const label = { running: '构建中', done: '完成', failed: '失败', canceled: '已取消' }[j.state] || j.state;
  $('conv-state').textContent = `${label} · ${j.output.split(/[\\/]/).pop()} · ${Math.round(j.seconds || (Date.now() - Date.parse(j.started_at)) / 1000)} s` + (j.error ? ' · ' + j.error : '');
  log.hidden = false;
  log.textContent = (j.tail || []).join('\n');
  log.scrollTop = log.scrollHeight;
  $('btn-convert').textContent = j.state === 'running' ? '取消构建' : '构建 TensorRT 引擎';
  return j.state === 'running';
}

$('btn-rescan').addEventListener('click', (e) => busy(e.target, () => refreshModels(true)));
$('btn-convert').addEventListener('click', (e) => busy(e.target, async () => {
  if (e.target.textContent === '取消构建') { await api('POST', '/convert/cancel'); }
  else { const j = await api('POST', '/convert', {}); toast('开始构建 ' + j.output + '，通常需要 5–20 分钟'); }
  await refreshConvert();
}));
$('btn-auto').addEventListener('click', (e) => busy(e.target, async () => { await api('POST', '/load/auto'); await refreshAll(); toast('加载完成'); }));
$('btn-unload').addEventListener('click', (e) => busy(e.target, async () => { await api('POST', '/engine/unload'); await refreshAll(); }));

// ── try ─────────────────────────────────────────────────────────────────────
$('try').addEventListener('submit', (e) => {
  e.preventDefault();
  const f = e.target.elements;
  busy(e.submitter, async () => {
    let questions;
    try { questions = JSON.parse(f.questions.value); } catch (err) { throw new Error('问题 JSON 无法解析：' + err.message); }
    const r = await api('POST', '/predict', { state: f.state.value, questions });
    $('try-time').textContent = `${r.timing.total_ms} ms（推理 ${r.timing.inference_ms} ms）`;
    $('try-out').textContent = JSON.stringify(r.answers, null, 2);
  });
});

// ── loop ────────────────────────────────────────────────────────────────────
async function refreshAll() { await Promise.all([refreshStatus(), refreshModels(false), refreshConvert()]); }
function poll(n) { let i = 0; const t = setInterval(async () => { await refreshStatus(); if (++i >= n) clearInterval(t); }, 1500); }

(async () => {
  try { await loadConfig(); } catch (e) { toast(e.message, true); }
  await refreshAll().catch((e) => toast(e.message, true));
  setInterval(async () => { const running = await refreshConvert().catch(() => false); if (running) await refreshStatus(); }, 3000);
  setInterval(() => refreshStatus(), 10000);
})();
