const $ = s => document.querySelector(s);
const CHAT_TRANSCRIPT_KEY = 'nm_chat_transcript';
const CHAT_HISTORY_KEY = 'nm_chat_history';
const ACTIVE_MEMORY_KEY = 'nm_active_memories';
const CONTEXT_USAGE_KEY = 'nm_context_usage';
const CONTEXT_SESSION_KEY = 'nm_context_session';
const SESSION_MEMORY_KEY = 'nm_session_memory_enabled';
const CHAT_TRANSCRIPT_LIMIT = 200;
const MAX_HISTORY_MESSAGE_CHARS = 20000; // must match config.MAX_MESSAGE_CHARS
async function api(p, o) {
  let r;
  try {
    r = await fetch(p, o);
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    const served = ['http:', 'https:'].includes(window.location.protocol);
    const hint = served
      ? 'Could not reach the Neural Map server. Make sure server.py is still running and reload the page from the same host.'
      : 'Open the web UI from http://127.0.0.1:8000 after starting server.py; opening web/index.html directly cannot reach /api.';
    throw new Error(`${hint} (${err.message || 'network request failed'})`);
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    let detail = data.detail;
    if (Array.isArray(detail)) detail = detail.map(d => (d && d.msg) ? d.msg : JSON.stringify(d)).join('; ');
    else if (detail && typeof detail === 'object') detail = JSON.stringify(detail, null, 2);
    throw new Error(detail || r.statusText);
  }
  return data;
}
let history = [];
let chatTranscript = [];
let activeMemories = [];
let nextTranscriptId = 1;
const HISTORY_LIMIT = 12; // six completed prior turns; must match server HISTORY_TURNS*2
let managerOpen = false;
let stateOpen = false;
let systemsOpen = false;
let observabilityOpen = false;
let actionsOpen = false;
let safetyOpen = false;
let pendingActionCount = 0;
let lastTurnObservability = null;
let retrievalDefaults = { top_k: 5, semantic: 1, keyword: 0.5, recency: 0.3, importance: 1, graph: 0.6, sensory: 0.7 };
let sessionMemoryEnabled = sessionStorage.getItem(SESSION_MEMORY_KEY) !== '0';

function selectedModel() {
  return $('#model').value || null;
}

function selectedInitiativeMode() {
  return $('#initiativeMode').value || 'helpful';
}

function modelQuery() {
  const model = selectedModel();
  return model ? '?model=' + encodeURIComponent(model) : '';
}

function initiativeQuery() {
  const params = new URLSearchParams();
  params.set('mode', selectedInitiativeMode());
  if (selectedModel()) params.set('model', selectedModel());
  return '?' + params.toString();
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function formatAssistantDisplayText(text) {
  const value = text == null ? '' : String(text);
  const parts = value.split(/(```[\s\S]*?(?:```|$))/);
  for (let i = 0; i < parts.length; i += 2) {
    let prose = parts[i].replace(/\r\n?/g, '\n');
    prose = prose.replace(/[ \t]+(?=#{1,6}[ \t]+\S)/g, '\n\n');
    prose = prose.replace(/^[ \t]*#{1,6}[ \t]+/gm, '');
    prose = prose.replace(/[ \t]+---[ \t]+/g, '\n\n');
    prose = prose.replace(
      /[ \t]+(?=(?:Audit Findings(?:[ \t]*\([^\n)]{1,100}\))?|Current Plan|Next Steps|Verdict(?:[ \t]+on[^:\n]{1,100})?):)/gi,
      '\n\n'
    );
    prose = prose.replace(
      /((?:^|\n)(?:Phase|Step)[ \t]+\d+(?:\.\d+)?[^\n]{0,180}?\bStatus:[ \t]*(?:IN PROGRESS|COMPLETE|COMPLETED|FAILED|BLOCKED|PAUSED|PASS|SECURE)(?:[ \t]*\([^\n)]{1,80}\))?)[ \t]+(?=[A-Z0-9])/gi,
      '$1\n\n'
    );
    prose = prose.replace(/((?:^|\n)Next Steps:)[ \t]+(?=(?:Phase|Step)\b)/gi, '$1\n\n');
    prose = prose.replace(
      /((?:^|\n)(?:Phase|Step)[ \t]+\d+(?:\.\d+)?(?:[ \t]*\([^\n)]{1,80}\))?[ \t]*(?:—|-)[ \t]*[A-Z][^.!?\n]{1,100}?)[ \t]+(?=(?:The|This|I|We|Now)\b)/gi,
      '$1\n\n'
    );
    prose = prose.replace(/([.!?:;)])[ \t]+(?=\d{1,2}[.)][ \t]+[A-Z0-9])/g, '$1\n\n');
    prose = prose.replace(/\*\*([^*\n]+)\*\*/g, '$1');
    prose = prose.replace(/^[ \t]*(?:[-+*]|•)[ \t]+/gm, '- ');
    prose = prose.split('\n').map(line => {
      if (line.length <= 180 || /^\s*(?:[-•]|\d{1,2}[.)])\s/.test(line)) return line;
      return line.replace(/([.!?])[ \t]+(?=["'“‘(]*[A-Z0-9])/g, '$1\n\n');
    }).join('\n');
    parts[i] = prose.trim();
  }
  return parts.filter(part => part.trim()).map(part => part.replace(/^\n+|\n+$/g, '')).join('\n\n')
    .replace(/\n{3,}/g, '\n\n').trim();
}

function setBubbleText(bubble, text) {
  const value = text == null ? '' : String(text);
  const displayValue = bubble.dataset.role === 'assistant'
    ? formatAssistantDisplayText(value)
    : value;
  const content = bubble.querySelector('.bubble-content');
  if (content) content.textContent = displayValue;
  else bubble.textContent = displayValue;
  bubble.dataset.copyText = displayValue;
  updateTranscriptBubble(bubble, value);
}

function saveTranscript() {
  try {
    localStorage.setItem(CHAT_TRANSCRIPT_KEY, JSON.stringify(chatTranscript.slice(-CHAT_TRANSCRIPT_LIMIT)));
  } catch (e) {}
}

function saveHistory() {
  try {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(cleanHistory(history).slice(-HISTORY_LIMIT)));
  } catch (e) {}
}

function rememberAssistantEvent(content) {
  const text = String(content || '').trim();
  if (!text) return;
  history = cleanHistory(history);
  // Action-panel decisions happen outside send(), so they must be attached to
  // an existing user turn explicitly. Otherwise the next request only sees the
  // assistant's stale "pending approval" reply and proposes the action again.
  if (!history.length) return;
  history.push({ role: 'assistant', content: text });
  history = cleanHistory(history).slice(-HISTORY_LIMIT);
  saveHistory();
}

function updateTranscriptBubble(bubble, content) {
  const id = Number(bubble && bubble.dataset ? bubble.dataset.transcriptId : 0);
  if (!id) return;
  const row = chatTranscript.find(item => item.id === id);
  if (!row) return;
  row.content = String(content || '');
  saveTranscript();
}

function recordBubble(bubble, role, content) {
  if (!bubble) return;
  const id = Number(bubble.dataset.transcriptId || nextTranscriptId++);
  bubble.dataset.transcriptId = String(id);
  const row = { id, role, content: String(content || '') };
  const existing = chatTranscript.findIndex(item => item.id === id);
  if (existing >= 0) chatTranscript[existing] = row;
  else chatTranscript.push(row);
  if (chatTranscript.length > CHAT_TRANSCRIPT_LIMIT) {
    chatTranscript = chatTranscript.slice(-CHAT_TRANSCRIPT_LIMIT);
  }
  nextTranscriptId = Math.max(nextTranscriptId, id + 1);
  saveTranscript();
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.left = '-9999px';
  document.body.appendChild(area);
  area.select();
  const ok = document.execCommand('copy');
  area.remove();
  if (!ok) throw new Error('copy failed');
}

async function copyBubble(button) {
  const bubble = button.closest('.bubble');
  const text = bubble ? bubble.dataset.copyText || '' : '';
  if (!text) return;
  const original = button.textContent;
  try {
    await copyText(text);
    button.textContent = '✓';
    button.classList.add('copied');
    setTimeout(() => {
      button.textContent = original;
      button.classList.remove('copied');
    }, 900);
  } catch (err) {
    button.textContent = '!';
    setTimeout(() => { button.textContent = original; }, 900);
  }
}

async function loadModels() {
  let data;
  try {
    data = await api('/api/models');
  } catch (e) {
    data = { models: [], default: localStorage.getItem('nm_model') || '' };
  }
  const sel = $('#model');
  sel.innerHTML = '';
  const fallback = data.default || localStorage.getItem('nm_model') || '';
  const models = [...new Set([...(data.models || []), fallback].filter(Boolean))];
  models.forEach(m => { const o = el('option', null, m); o.value = m; sel.appendChild(o); });
  const saved = localStorage.getItem('nm_model');
  sel.value = (saved && models.includes(saved)) ? saved : fallback;
  onModelChange();
}

async function onModelChange() {
  const name = $('#model').value;
  if (!name) { $('#ctxhint').textContent = ''; return; }
  const previousModel = localStorage.getItem('nm_model');
  localStorage.setItem('nm_model', name);
  if (previousModel && previousModel !== name) clearContextUsage();
  else restoreContextUsage(name);
  $('#ctxhint').textContent = '…';
  const info = await api('/api/model_info?name=' + encodeURIComponent(name));
  if (info.max_context) {
    $('#ctx').max = info.max_context;
    $('#ctxhint').textContent = 'max ' + info.max_context.toLocaleString();
  } else {
    $('#ctxhint').textContent = '';
  }
  loadStats().catch(() => {});
  loadRetrievalScopes().then(syncRetrievalLabels).catch(() => {});
  loadInitiatives().catch(() => {});
  if (managerOpen) loadMemories().catch(() => {});
}

function clearContextUsage() {
  const usage = $('#ctxusage');
  const meter = $('#ctxmeter');
  if (usage) {
    usage.textContent = 'usage after next response';
    usage.title = 'Peak input prompt usage will appear after a completed turn.';
  }
  if (meter) {
    meter.value = 0;
    meter.classList.remove('context-warn', 'context-high');
  }
  const session = $('#ctxsession');
  if (session) {
    session.textContent = '';
    session.title = 'Cumulative input tokens processed this session.';
  }
  try { localStorage.removeItem(CONTEXT_USAGE_KEY); } catch (e) {}
  try { localStorage.removeItem(CONTEXT_SESSION_KEY); } catch (e) {}
}

function restoreContextUsage(model) {
  try {
    const saved = JSON.parse(localStorage.getItem(CONTEXT_USAGE_KEY) || 'null');
    if (saved && saved.model === model) renderContextUsage(saved, false);
    else clearContextUsage();
  } catch (e) {
    clearContextUsage();
  }
  renderContextSession(model);
}

function readContextSession() {
  try {
    return JSON.parse(localStorage.getItem(CONTEXT_SESSION_KEY) || 'null');
  } catch (e) {
    return null;
  }
}

function renderContextSession(model) {
  const span = $('#ctxsession');
  if (!span) return;
  const saved = readContextSession();
  if (!saved || !saved.total_tokens || (model && saved.model !== model)) {
    span.textContent = '';
    span.title = 'Cumulative input tokens processed this session.';
    return;
  }
  span.textContent = `Σ ${Number(saved.total_tokens).toLocaleString()} session`;
  span.title = `Cumulative input tokens processed this session: ${Number(saved.total_tokens).toLocaleString()} across ${saved.turns} turn(s), ${saved.calls} model call(s). `
    + 'Prompts are rebuilt every turn, so this total keeps growing even though per-turn occupancy plateaus once the history window is full. '
    + 'Resets with /clear, /purge, or a model change.';
}

function accumulateContextSession(data) {
  if (!data) return;
  const turnTokens = Math.max(0, Number(data.turn_total_tokens) || Number(data.used_tokens) || 0);
  if (!turnTokens) return;
  const saved = readContextSession();
  const base = (saved && saved.model === data.model)
    ? saved
    : { model: data.model, total_tokens: 0, turns: 0, calls: 0 };
  base.total_tokens = Math.max(0, Number(base.total_tokens) || 0) + turnTokens;
  base.turns = Math.max(0, Number(base.turns) || 0) + 1;
  base.calls = Math.max(0, Number(base.calls) || 0) + Math.max(0, Number(data.calls) || 0);
  try { localStorage.setItem(CONTEXT_SESSION_KEY, JSON.stringify(base)); } catch (e) {}
  renderContextSession(data.model);
}

function renderContextUsage(data, persist = true) {
  if (!data || !data.configured_tokens) return;
  const used = Math.max(0, Number(data.used_tokens) || 0);
  const limit = Math.max(1, Number(data.configured_tokens) || 1);
  const percent = Math.max(0, (Number(data.percent) || (used / limit * 100)));
  const measurement = data.measurement === 'exact' ? 'exact' : 'estimated';
  const usage = $('#ctxusage');
  const meter = $('#ctxmeter');
  usage.textContent = `${used.toLocaleString()} / ${limit.toLocaleString()} · ${percent.toFixed(1)}% ${measurement}`;
  usage.title = `${data.scope || 'Peak input prompt for this turn'}; ${Math.max(0, limit - used).toLocaleString()} tokens remaining; ${data.calls || 0} model call(s).`;
  meter.value = Math.min(100, percent);
  meter.classList.toggle('context-warn', percent >= 70 && percent < 90);
  meter.classList.toggle('context-high', percent >= 90);
  if (persist) {
    try { localStorage.setItem(CONTEXT_USAGE_KEY, JSON.stringify(data)); } catch (e) {}
  }
}

function markContextUsageUnavailable() {
  const usage = $('#ctxusage');
  const meter = $('#ctxmeter');
  usage.textContent = 'usage unavailable · restart server';
  usage.title = 'This chat response did not include context_usage telemetry. Restart Neural Map so the Python backend matches the UI.';
  meter.value = 0;
  meter.classList.remove('context-warn', 'context-high');
}

function setStats(n) { $('#stats').textContent = n + (n === 1 ? ' memory' : ' memories'); }

async function loadStats() {
  const s = await api('/api/stats' + modelQuery());
  setStats(s.count);
}

function clampNum(value, min, max, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}

function retrievalInputs() {
  return {
    top_k: $('#topK'),
    semantic: $('#wSemantic'),
    keyword: $('#wKeyword'),
    recency: $('#wRecency'),
    importance: $('#wImportance'),
    graph: $('#wGraph'),
    sensory: $('#wSensory'),
    category: $('#wCategory')
  };
}

function setRetrieval(settings) {
  const inputs = retrievalInputs();
  inputs.top_k.value = settings.top_k;
  inputs.semantic.value = settings.semantic;
  inputs.keyword.value = settings.keyword;
  inputs.recency.value = settings.recency;
  inputs.importance.value = settings.importance;
  inputs.graph.value = settings.graph;
  inputs.sensory.value = settings.sensory;
  inputs.category.value = settings.category || '';
  syncRetrievalLabels();
}

function retrievalSettings() {
  const inputs = retrievalInputs();
  const s = {
    top_k: Math.round(clampNum(inputs.top_k.value, 1, 25, retrievalDefaults.top_k)),
    semantic: clampNum(inputs.semantic.value, 0, 5, retrievalDefaults.semantic),
    keyword: clampNum(inputs.keyword.value, 0, 5, retrievalDefaults.keyword),
    recency: clampNum(inputs.recency.value, 0, 5, retrievalDefaults.recency),
    importance: clampNum(inputs.importance.value, 0, 5, retrievalDefaults.importance),
    graph: clampNum(inputs.graph.value, 0, 5, retrievalDefaults.graph),
    sensory: clampNum(inputs.sensory.value, 0, 5, retrievalDefaults.sensory),
    category: inputs.category.value || null
  };
  // Every signal now produces candidates (semantic/keyword/graph/sensory, plus the
  // recency/importance pulls), so only an all-zero request is useless — restore
  // semantic in that one case so a search always has something to find with.
  if (s.semantic + s.keyword + s.graph + s.sensory + s.recency + s.importance <= 0) {
    s.semantic = retrievalDefaults.semantic || 1;
  }
  return s;
}

function syncRetrievalLabels() {
  const s = retrievalSettings();
  $('#topK').value = s.top_k;
  $('#wSemantic').value = s.semantic;
  $('#wKeyword').value = s.keyword;
  $('#wRecency').value = s.recency;
  $('#wImportance').value = s.importance;
  $('#wGraph').value = s.graph;
  $('#wSensory').value = s.sensory;
  $('#topKVal').textContent = String(s.top_k);
  $('#wSemanticVal').textContent = s.semantic.toFixed(1);
  $('#wKeywordVal').textContent = s.keyword.toFixed(1);
  $('#wRecencyVal').textContent = s.recency.toFixed(1);
  $('#wImportanceVal').textContent = s.importance.toFixed(1);
  $('#wGraphVal').textContent = s.graph.toFixed(1);
  $('#wSensoryVal').textContent = s.sensory.toFixed(1);
  localStorage.setItem('nm_retrieval', JSON.stringify(s));
}

async function loadRetrievalScopes() {
  try {
    const data = await api('/api/categories' + modelQuery());
    populateSelect($('#wCategory'), data.categories || [], 'All categories');
  } catch (e) {}
}

async function loadRetrievalDefaults() {
  try {
    retrievalDefaults = await api('/api/retrieval_defaults');
  } catch (e) {}
  await loadRetrievalScopes();
  const saved = localStorage.getItem('nm_retrieval');
  try {
    setRetrieval(saved ? { ...retrievalDefaults, ...JSON.parse(saved) } : retrievalDefaults);
  } catch (e) {
    setRetrieval(retrievalDefaults);
  }
}

function addBubble(role, text, opts = {}) {
  const wrap = el('div', 'msg ' + role);
  const bubble = el('div', 'bubble');
  bubble.dataset.role = role;
  bubble.appendChild(el('div', 'bubble-content'));
  const copy = el('button', 'copy-bubble', '⧉');
  copy.type = 'button';
  copy.title = 'Copy message';
  copy.setAttribute('aria-label', 'Copy message');
  copy.addEventListener('click', () => copyBubble(copy));
  bubble.appendChild(copy);
  const persist = opts.persist !== false;
  if (persist) bubble.dataset.transcriptId = String(nextTranscriptId++);
  setBubbleText(bubble, text);
  if (persist) recordBubble(bubble, role, text);
  wrap.appendChild(bubble);
  $('#messages').appendChild(wrap);
  $('#messages').scrollTop = $('#messages').scrollHeight;
  return wrap.querySelector('.bubble');
}

function memoryLabel(m) {
  const category = m.category || 'memory';
  const date = (m.date || '').slice(0, 10);
  return date ? `${category} · ${date}` : category;
}

function saveActiveMemories() {
  try {
    localStorage.setItem(ACTIVE_MEMORY_KEY, JSON.stringify(activeMemories.slice(0, 10)));
  } catch (e) {}
}

function renderActiveMemories() {
  const box = $('#activeMemories');
  if (!box) return;
  box.innerHTML = '';
  if (!activeMemories.length) {
    box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');
  box.appendChild(el('div', 'active-title', 'Activated'));
  activeMemories.forEach(m => {
    const chip = el('div', 'active-memory');
    const text = el('button', 'active-memory-text', `${memoryLabel(m)}: ${(m.text || '').slice(0, 96)}`);
    text.type = 'button';
    text.title = m.text || '';
    text.addEventListener('click', () => showMemoryDetail(m.id));
    const remove = el('button', 'active-remove', 'x');
    remove.type = 'button';
    remove.title = 'Remove from active chat context';
    remove.addEventListener('click', () => deactivateMemory(m.id));
    chip.appendChild(text);
    chip.appendChild(remove);
    box.appendChild(chip);
  });
}

function activateMemory(memory) {
  if (!memory || !memory.id) return;
  const existing = activeMemories.findIndex(m => m.id === memory.id);
  const item = {
    id: memory.id,
    text: memory.text || '',
    category: memory.category || 'memory',
    date: memory.date || '',
    source: memory.source || null,
    pinned: !!memory.pinned
  };
  if (existing >= 0) activeMemories.splice(existing, 1);
  activeMemories.unshift(item);
  activeMemories = activeMemories.slice(0, 10);
  saveActiveMemories();
  renderActiveMemories();
  addBubble('assistant', `Recollected for the conversation:\n${memoryLabel(item)}\n${item.text}`, { persist: false });
}

function deactivateMemory(id) {
  activeMemories = activeMemories.filter(m => m.id !== id);
  saveActiveMemories();
  renderActiveMemories();
}

function renderRecalled(mems) {
  const box = $('#recalled');
  box.innerHTML = '';
  if (!mems || !mems.length) { box.appendChild(el('p', 'empty', 'No memories matched this turn.')); return; }
  const max = Math.max(...mems.map(m => m.score)) || 1;
  mems.forEach(m => {
    const card = el('div', 'card');
    card.style.cursor = (m.id && !m.sensory) ? 'pointer' : 'default';
    if (m.id && !m.sensory) card.addEventListener('click', () => showMemoryDetail(m.id));
    card.appendChild(el('div', 'card-text', m.text));
    if (m.id && !m.sensory) {
      const actions = el('div', 'card-actions');
      const recall = el('button', 'ghost mini', 'Recall');
      recall.type = 'button';
      recall.title = 'Activate this memory in the chat context';
      recall.addEventListener('click', event => {
        event.stopPropagation();
        activateMemory(m);
      });
      actions.appendChild(recall);
      card.appendChild(actions);
    }
    const bar = el('div', 'bar'); const fill = el('div', 'fill');
    fill.style.width = (100 * m.score / max) + '%'; bar.appendChild(fill); card.appendChild(bar);
    const origin = sourceLabel(m.source) || m.category;
    const pinned = m.pinned ? 'pinned · ' : '';
    const sensory = m.sensory ? `${m.sensory_kind || 'sensory'} · sense ${m.sensory_similarity} · ` : '';
    card.appendChild(el('div', 'card-meta',
      `${m.activated ? 'activated · ' : ''}score ${m.score} · sim ${m.similarity} · kw ${m.keyword} · rec ${m.recency} · ${pinned}${m.graph_linked ? 'graph · ' : ''}${sensory}${origin}`));
    box.appendChild(card);
  });
}

function renderReminders(list) {
  const box = $('#reminders');
  box.innerHTML = '';
  if (!list || !list.length) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  box.appendChild(el('span', 'rem-title', 'Proactive reminders'));
  list.forEach(r => {
    const chip = el('div', 'rem-chip' + (r.overdue ? ' overdue' : ''));
    chip.appendChild(el('span', 'due', (r.overdue ? '⚠ ' : '') + (r.due || '')));
    chip.appendChild(el('span', 'action', (r.text || '').replace(/^\[due [^\]]*\]\s*/, '')));
    box.appendChild(chip);
  });
}

function renderInitiatives(list) {
  const box = $('#initiative');
  box.innerHTML = '';
  if (!list || !list.length) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  box.appendChild(el('span', 'init-title', 'Initiative'));
  list.forEach(item => {
    const chip = el('div', 'init-chip');
    const decisionLabel = item.experimental ? `${item.decision_mode || 'intuitive'} experiment` : item.decision_mode;
    chip.appendChild(el('span', 'kind', [decisionLabel, item.kind || 'nudge'].filter(Boolean).join(' · ')));
    chip.appendChild(el('span', 'text', item.text || ''));
    if (item.will_decision_id && !item.will_outcome) {
      const useful = el('button', null, 'Useful');
      useful.type = 'button';
      useful.title = 'Reinforce similar Will suggestions';
      useful.addEventListener('click', () => recordWillOutcome(item, 'useful'));
      chip.appendChild(useful);
      const interesting = el('button', null, 'Interesting');
      interesting.type = 'button';
      interesting.title = 'Mark this as an interesting experimental result';
      interesting.addEventListener('click', () => recordWillOutcome(item, 'interesting'));
      chip.appendChild(interesting);
    } else if (item.will_outcome) {
      chip.appendChild(el('span', 'kind', `rated ${item.will_outcome}`));
    }
    if (item.admissibility === 'needs_evidence') {
      const blocked = el('button', null, 'Evidence needed');
      blocked.type = 'button';
      blocked.disabled = true;
      blocked.title = 'Grounding evidence or explicit confirmation is required before packaging this initiative.';
      chip.appendChild(blocked);
    } else {
      const packageBtn = el('button', null, 'Package');
      packageBtn.type = 'button';
      packageBtn.title = 'Prepare a reviewable Action Package';
      packageBtn.addEventListener('click', () => packageInitiative(item));
      chip.appendChild(packageBtn);
    }
    const dismiss = el('button', null, 'Dismiss');
    dismiss.type = 'button';
    dismiss.addEventListener('click', () => dismissInitiative((item.source_ids && item.source_ids.length) ? item.source_ids : [item.id]));
    chip.appendChild(dismiss);
    box.appendChild(chip);
  });
}

function pct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'n/a';
  return Math.round(Math.max(0, Math.min(1, n)) * 100) + '%';
}

function renderAwareness(data, attention) {
  const box = $('#awareness');
  box.innerHTML = '';
  if (!data || data.enabled === false) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  const main = el('div', 'awareness-main');
  main.appendChild(el('span', 'awareness-title', 'Awareness'));
  main.appendChild(el('span', 'aware-focus', data.current_focus || 'idle'));
  main.appendChild(el('span', 'aware-chip', data.stance || 'neutral'));
  main.appendChild(el('span', 'aware-chip', `energy ${pct(data.energy)}`));
  main.appendChild(el('span', 'aware-chip', `confidence ${pct(data.confidence)}`));
  if (attention && attention.enabled !== false) {
    main.appendChild(el('span', 'aware-chip', `attention ${attention.focused_domain || 'general'}`));
    (attention.domains ? Object.entries(attention.domains) : [])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .filter(([, score]) => Number(score) > 0)
      .forEach(([domain, score]) => main.appendChild(el('span', 'aware-chip', `${domain} ${pct(score)}`)));
  }
  const loops = (data.open_loops || []).length;
  if (loops) main.appendChild(el('span', 'aware-chip', `${loops} open loop${loops === 1 ? '' : 's'}`));
  box.appendChild(main);
  box.appendChild(el('div', 'aware-next', data.next_best_action || 'wait for the next request'));
}

async function loadAwareness() {
  try {
    const [aware, attn] = await Promise.all([
      api('/api/awareness'),
      api('/api/attention').catch(() => null)
    ]);
    renderAwareness(aware, attn);
  } catch (e) {
    renderAwareness(null);
  }
}

async function loadReminders() {
  try {
    const data = await api('/api/reminders' + initiativeQuery());
    renderReminders(data.reminders);
  } catch (e) { /* non-fatal */ }
}

async function loadInitiatives() {
  try {
    const data = await api('/api/initiative' + initiativeQuery());
    renderInitiatives(data.initiatives);
    if (data.initiatives && data.initiatives.length) {
      await api('/api/initiative/contact', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: data.initiatives.map(i => i.id), mode: selectedInitiativeMode(), model: selectedModel() })
      }).catch(() => {});
    }
  } catch (e) { /* non-fatal */ }
}

async function dismissInitiative(ids) {
  try {
    const requested = Array.isArray(ids) ? ids : [ids];
    await api('/api/initiative/dismiss', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: requested, model: selectedModel() })
    });
    await loadInitiatives();
  } catch (e) {
    $('#initiative').prepend(el('span', 'empty', e.message));
  }
}

async function recordWillOutcome(item, outcome) {
  try {
    const target = item.will_initiative_id || item.id;
    const result = await api('/api/will/outcome', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ initiative_id: target, outcome, model: selectedModel() })
    });
    await loadInitiatives();
    addBubble('assistant', `Will learned from that ${outcome} result:\n${result.reflection || ''}`);
  } catch (e) {
    $('#initiative').prepend(el('span', 'empty', e.message));
  }
}

async function packageInitiative(item) {
  try {
    const res = await api('/api/action_packages/from_initiative', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initiative_id: item.id || null,
        text: item.text || '',
        kind: item.kind || null,
        priority: item.priority || 4,
        source_ids: item.source_ids || [],
        model: selectedModel()
      })
    });
    await loadPendingActions();
    openActions();
    addBubble('assistant', res.existing ? 'That Action Package is already pending review.' : 'Prepared an Action Package for your review.');
  } catch (e) {
    $('#initiative').prepend(el('span', 'empty', e.message));
  }
}

function formatBriefing(data) {
  const lines = ['Briefing'];
  if (data.reminders && data.reminders.length) {
    lines.push('', 'Due reminders:');
    data.reminders.slice(0, 5).forEach(r => lines.push(`- ${(r.due || '').slice(0, 10)} ${String(r.text || '').replace(/^\[due [^\]]*\]\s*/, '')}`));
  }
  if (data.conflicts && data.conflicts.length) {
    lines.push('', 'Conflicts to resolve:');
    data.conflicts.slice(0, 3).forEach(i => lines.push(`- ${i.text}`));
  }
  if (data.open_loops && data.open_loops.length) {
    lines.push('', 'Open loops:');
    data.open_loops.slice(0, 3).forEach(i => lines.push(`- ${i.text}`));
  }
  if (data.preferences && data.preferences.length) {
    lines.push('', 'Style preferences I am using:');
    data.preferences.slice(0, 3).forEach(p => lines.push(`- ${p.text}`));
  }
  if (data.experiment_reflections && data.experiment_reflections.length) {
    lines.push('', 'Interesting Will experiments:');
    data.experiment_reflections.slice(0, 3).forEach(row => lines.push(`- ${row.reflection}`));
  }
  if (data.suggested_next_action) {
    lines.push('', 'Suggested next action:', data.suggested_next_action.replace(/^\[due [^\]]*\]\s*/, ''));
  }
  if (lines.length === 1) lines.push('', 'Nothing needs attention right now.');
  return lines.join('\n');
}

async function showBriefing() {
  try {
    const data = await api('/api/briefing' + initiativeQuery());
    renderInitiatives(data.initiatives);
    addBubble('assistant', formatBriefing(data));
  } catch (e) {
    addBubble('assistant', 'Briefing failed: ' + e.message);
  }
}

async function runWillReview() {
  const btn = $('#willBtn'); const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Thinking…';
  try {
    const res = await api('/api/will', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selectedModel(), mode: selectedInitiativeMode() })
    });
    renderInitiatives(res.active_initiatives || []);
    if (res.created) {
      const note = res.notification || {};
      const decision = res.decision || {};
      const label = decision.experimental ? `${decision.selected_mode || 'intuitive'} experiment` : (decision.selected_mode || 'logical');
      addBubble('assistant', `Will review created a ${label} nudge:\n${note.message || ''}\n\nSelection: ${decision.selection_reason || 'highest-scoring candidate'}`);
    } else {
      addBubble('assistant', 'Will review: ' + (res.reason || 'nothing worth surfacing right now.'));
    }
  } catch (e) {
    addBubble('assistant', 'Will review failed: ' + e.message);
  }
  btn.disabled = false; btn.textContent = label;
}

function topEntries(obj, limit = 5) {
  return Object.entries(obj || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([k, v]) => `${k}: ${v}`)
    .join(', ') || 'none';
}

function formatTraceMetrics(trace) {
  const metrics = trace.metrics || {};
  const complexity = metrics.complexity_score ?? (trace.events || []).length;
  const latency = Number(metrics.latency_delta_ms);
  const parts = [`complexity ${complexity}`];
  if (Number.isFinite(latency)) parts.push(`${latency.toFixed(0)}ms`);
  return parts.join(' · ');
}

function compactToolResult(call) {
  const result = call.result || {};
  if (result.error) return `${call.tool}: ${result.error}`;
  if (call.tool === 'web_search') {
    const count = (result.results || []).length;
    const provider = result.provider || 'web';
    const errors = (result.errors || []).map(e => e.error || JSON.stringify(e)).filter(Boolean);
    return `${call.tool}: ${count} result(s) from ${provider}${errors.length ? ` · warning ${errors.join('; ')}` : ''}`;
  }
  if (result.pending && result.action) return `${call.tool}: pending approval ${result.action.id}`;
  return `${call.tool}: ok`;
}

function traceEventText(event) {
  const fields = event.fields || {};
  const extras = Object.entries(fields)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .slice(0, 4)
    .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(',') : v}`)
    .join(' · ');
  return `${event.layer}:${event.status}${extras ? ' · ' + extras : ''}`;
}

function collectTurnObservability(res, toolResults, usedTools, pendingTools) {
  const webData = res.web_data || {};
  const trace = res.trace || null;
  const webTool = (toolResults || []).find(call => call.tool === 'web_search' && call.result);
  const web = webTool ? webTool.result : webData;
  return {
    at: new Date().toLocaleTimeString(),
    trace,
    thinking: res.thinking || null,
    tools: usedTools || [],
    pending: pendingTools || [],
    toolResults: toolResults || [],
    web: web && Object.keys(web).length ? web : null,
    awareness: res.awareness || null,
    attention: res.attention || null,
    workspace: res.workspace || null,
    grounding: res.grounding || null,
    groundingGate: res.grounding_gate || null,
    claimGuard: res.claim_guard || null,
    presence: res.presence || null,
    contextUsage: res.context_usage || null,
    council: res.council || null,
  };
}

function renderObservabilityTurn(turn) {
  const list = $('#observabilityLive');
  if (!list) return;
  list.innerHTML = '';
  if (!turn) {
    list.appendChild(el('p', 'empty', 'No chat turn observed yet.'));
    return;
  }
  const summary = el('div', 'state-row');
  summary.appendChild(el('div', 'state-text', `Last turn ${turn.at}`));
  const bits = [];
  if (turn.trace) bits.push(`trace ${turn.trace.id}`, formatTraceMetrics(turn.trace));
  if (turn.thinking && turn.thinking.path) bits.push(turn.thinking.path);
  if (turn.tools.length) bits.push(`tools ${turn.tools.join(', ')}`);
  if (turn.pending.length) bits.push(`pending ${turn.pending.join(', ')}`);
  if (turn.workspace && turn.workspace.enabled !== false) {
    bits.push(`integration proxy ${((turn.workspace.metrics || {}).integration_proxy || 0)}`);
  }
  if (turn.grounding && turn.grounding.enabled !== false) {
    bits.push(`grounded ${((turn.grounding.metrics || {}).concept_count || 0)} concepts`);
  }
  if (turn.groundingGate && turn.groundingGate.enabled !== false) {
    bits.push(`grounding ${turn.groundingGate.epistemic_posture || 'unknown'}`);
  }
  if (turn.claimGuard) {
    bits.push(`claim guard ${turn.claimGuard.status || 'unknown'}`);
  }
  if (turn.presence && turn.presence.enabled !== false) {
    const social = (turn.presence.claims || []).find(row => row.claim === 'It constitutes a non-biological social presence');
    bits.push(`social presence ${(social || {}).status || 'not yet demonstrated'}`);
  }
  if (turn.contextUsage) {
    bits.push(`context ${Number(turn.contextUsage.percent || 0).toFixed(1)}% ${turn.contextUsage.measurement || 'estimated'}`);
  }
  summary.appendChild(stateMeta(bits));
  list.appendChild(summary);

  if (turn.web) {
    const row = el('div', 'state-row');
    const count = (turn.web.results || []).length;
    row.appendChild(el('div', 'state-text', `Web search: ${count} result(s) from ${turn.web.provider || 'provider'}`));
    const meta = [turn.web.query ? `query "${turn.web.query}"` : '', ...(turn.web.errors || []).map(e => e.error || JSON.stringify(e))].filter(Boolean);
    row.appendChild(stateMeta(meta));
    (turn.web.results || []).slice(0, 5).forEach(result => {
      row.appendChild(el('div', 'toolnote', `${result.title || 'Untitled'}${result.source ? ' · ' + result.source : ''}`));
    });
    list.appendChild(row);
  }

  if (turn.toolResults.length) {
    const row = el('div', 'state-row');
    row.appendChild(el('div', 'state-text', 'Tool results'));
    turn.toolResults.forEach(call => row.appendChild(el('div', 'toolnote', compactToolResult(call))));
    list.appendChild(row);
  }

  if (turn.trace && turn.trace.events && turn.trace.events.length) {
    const row = el('div', 'state-row');
    row.appendChild(el('div', 'state-text', 'Trace events'));
    turn.trace.events.forEach(event => row.appendChild(el('div', 'toolnote', traceEventText(event))));
    list.appendChild(row);
  }
}

function renderTraceList(traces) {
  const list = $('#observabilityTraceList');
  if (!list) return;
  list.innerHTML = '';
  if (!traces || !traces.length) {
    list.appendChild(el('p', 'empty', 'No traces recorded yet.'));
    return;
  }
  traces.forEach(trace => {
    const row = el('div', 'state-row');
    row.appendChild(el('div', 'state-text', `${trace.id} · ${trace.status || 'trace'}`));
    row.appendChild(stateMeta([trace.kind, trace.model, formatTraceMetrics(trace)]));
    (trace.events || []).slice(-8).forEach(event => row.appendChild(el('div', 'toolnote', traceEventText(event))));
    list.appendChild(row);
  });
}

async function loadObservability() {
  renderObservabilityTurn(lastTurnObservability);
  const data = await api('/api/traces?limit=10');
  renderTraceList(data.traces || []);
}

function openObservability() {
  observabilityOpen = true;
  $('#observabilityPanel').classList.remove('hidden');
  loadObservability().catch(err => { $('#observabilityTraceList').textContent = err.message; });
}

function closeObservability() {
  observabilityOpen = false;
  $('#observabilityPanel').classList.add('hidden');
}

async function showDiagnostics() {
  try {
    const [health, logs, traces, timeline, graphData] = await Promise.all([
      api('/api/health' + modelQuery()),
      api('/api/logs?limit=8'),
      api('/api/traces?limit=5'),
      api('/api/timeline' + modelQuery()),
      api('/api/graph' + modelQuery())
    ]);
    const lines = ['Diagnostics'];
    lines.push('', `Ollama: ${health.ollama.reachable ? 'reachable' : 'unreachable'}`);
    lines.push(`Chat model: ${health.chat_model.name} (${health.chat_model.available === null ? 'unknown' : health.chat_model.available ? 'available' : 'missing'})`);
    lines.push(`Embed model: ${health.embed_model.name} (${health.embed_model.available === null ? 'unknown' : health.embed_model.available ? 'available' : 'missing'})`);
    if (health.role_models) {
      const routed = Object.entries(health.role_models).map(([role, info]) => {
        const status = info.available === null ? 'unknown' : info.available ? 'available' : 'missing';
        return `${role}=${info.name} (${status})`;
      });
      lines.push(`Role models: ${routed.join(', ')}`);
    }
    lines.push(`Memories: ${health.memories.total} total, ${health.memories.vector} vector, ${health.memories.text_only} text-only, ${health.memories.pinned} pinned`);
    if (health.cognitive_state && health.cognitive_state.enabled) {
      const cs = health.cognitive_state;
      lines.push(`State: ${(cs.active_goals || []).length} goals, ${(cs.world_model || []).length} world facts, ${(cs.belief_ledger || []).length} beliefs`);
    }
    lines.push(`Source types: ${topEntries(health.memories.source_types)}`);
    lines.push(`Graph: ${health.graph.entities} entities, ${health.graph.edges} edges, ${health.graph.resolving_edges}/${health.graph.scoped_edges} resolving in this store`);
    lines.push(`Features: web ${health.features.web_search ? health.features.web_search_mode : 'off'}, context ${health.features.web_search_user_derived_with_context ? 'user-derived' : 'guarded'}, tools ${health.features.tool_calling ? 'on' : 'off'}, sleep ${health.features.sleep ? 'on' : 'off'}, will ${health.features.will ? 'on' : 'off'}`);
    if (health.sleep && health.sleep.last_run) lines.push(`Last sleep: ${String(health.sleep.last_run).slice(0, 19)} · archived ${health.sleep.archived || 0} · dreams ${health.sleep.dreamed || 0}`);
    if (health.will && health.will.last_run) {
      const result = health.will.last_result || {};
      const outcome = result.reason ? ` · ${result.status || 'completed'}: ${result.reason}` : '';
      const parsed = new Date(health.will.last_run);
      const when = Number.isNaN(parsed.getTime()) ? String(health.will.last_run) : parsed.toLocaleString();
      lines.push(`Last will review: ${when} · today ${health.will.outreach_today || 0}${outcome}`);
    }
    if (health.emotion) {
      const e = health.emotion;
      lines.push(`Mood: ${(e.labels || ['neutral']).join(', ')} · happiness ${e.happiness} · stress ${e.stress} · confidence ${e.confidence} · curiosity ${e.curiosity}`);
    }
    if (health.awareness && health.awareness.enabled !== false) {
      const a = health.awareness;
      lines.push(`Awareness: ${a.current_focus || 'idle'} · ${a.stance || 'neutral'} · next ${a.next_best_action || 'wait'}`);
    }
    if (health.self_model && health.self_model.enabled !== false) {
      const sm = health.self_model;
      lines.push(`Self model: ${(sm.current_projects || []).length} project(s), ${(sm.open_promises || []).length} promise(s), ${(sm.lessons_learned || []).length} lesson(s)`);
    }
    if (timeline.days && timeline.days.length) {
      lines.push('', 'Recent timeline:');
      timeline.days.slice(0, 5).forEach(d => lines.push(`- ${d.date}: ${d.count} memories (${topEntries(d.categories, 3)})`));
    }
    if (graphData.edges && graphData.edges.length) {
      lines.push('', 'Graph sample:');
      graphData.edges.slice(0, 5).forEach(e => lines.push(`- ${e.subject} — ${e.relation} -> ${e.object}`));
    }
    if (logs.events && logs.events.length) {
      lines.push('', 'Recent fallback events:');
      logs.events.forEach(e => lines.push(`- ${(e.time || '').slice(0, 19)} ${e.level} ${e.event}${e.error ? ': ' + e.error : ''}`));
    }
    if (traces.traces && traces.traces.length) {
      lines.push('', 'Recent cognitive traces:');
      traces.traces.forEach(t => {
        const tail = (t.events || []).slice(-4).map(e => `${e.layer}:${e.status}`).join(' > ');
        lines.push(`- ${t.id} ${t.status} · ${formatTraceMetrics(t)} · ${tail || 'no events'}`);
      });
    }
    addBubble('assistant', lines.join('\n'));
  } catch (e) {
    addBubble('assistant', 'Diagnostics failed: ' + e.message);
  }
}

async function resetEmotion() {
  if (!confirm('Reset affective stance to neutral?')) return;
  try {
    const e = await api('/api/emotion/reset', { method: 'POST' });
    addBubble('assistant', `Mood reset: ${(e.labels || ['neutral']).join(', ')}`);
  } catch (err) {
    addBubble('assistant', 'Mood reset failed: ' + err.message);
  }
}

function splitStateList(text) {
  return String(text || '').split(/[;\n]/).map(s => s.trim()).filter(Boolean);
}

function confidenceValue(sel, fallback) {
  return clampNum($(sel).value, 0, 1, fallback);
}

function stateMeta(parts) {
  return el('div', 'memory-meta', parts.filter(Boolean).join(' · '));
}

function actionButton(label, fn, cls = 'ghost') {
  const btn = el('button', cls, label);
  btn.type = 'button';
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      await fn();
    } catch (err) {
      addBubble('assistant', `${label} failed: ${err.message}`);
    } finally {
      btn.disabled = false;
    }
  });
  return btn;
}

function jsonBlock(value) {
  return el('div', 'preview-block', JSON.stringify(value || {}, null, 2));
}

function renderIssueList(parent, issues) {
  if (!issues || !issues.length) {
    parent.appendChild(el('div', 'toolnote', 'No issues reported.'));
    return;
  }
  issues.slice(0, 8).forEach(issue => {
    const text = `${issue.severity || 'issue'} · ${issue.code || 'unknown'}: ${issue.message || ''}`;
    parent.appendChild(el('div', issue.severity === 'error' ? 'memory-meta issue-error' : 'memory-meta', text));
  });
  if (issues.length > 8) parent.appendChild(el('div', 'memory-meta', `${issues.length - 8} more issue(s)`));
}

function renderIntegrity(data) {
  const list = $('#integrityList');
  list.innerHTML = '';
  if (!data) {
    list.appendChild(el('p', 'empty', 'Integrity status unavailable.'));
    return;
  }
  const row = el('div', 'state-row');
  const summary = data.summary || {};
  row.appendChild(el('div', 'state-text', data.ok ? 'Integrity checks passed.' : 'Integrity checks need attention.'));
  row.appendChild(stateMeta([`${summary.errors || 0} error(s)`, `${summary.warnings || 0} warning(s)`]));
  Object.entries(data.checks || {}).forEach(([name, check]) => {
    const block = el('div');
    block.appendChild(el('div', 'memory-meta', `${name} · ${check.count || 0} record(s)`));
    renderIssueList(block, check.issues || []);
    row.appendChild(block);
  });
  if (!data.unavailable) {
    const actions = el('div', 'state-actions');
    actions.appendChild(actionButton('Repair Graph', repairIntegrity, data.ok ? 'ghost' : 'danger ghost'));
    row.appendChild(actions);
  }
  list.appendChild(row);
}

function renderHomeostasisState(state, rulesData) {
  const list = $('#homeostasisList');
  list.innerHTML = '';
  if (!state) {
    list.appendChild(el('p', 'empty', 'Homeostasis state unavailable.'));
    return;
  }
  const row = el('div', 'state-row');
  row.appendChild(el('div', 'state-text', state.enabled === false ? 'Homeostasis disabled.' : 'Homeostasis active.'));
  const variables = state.variables || state.derived || {};
  const signals = state.last_signals || state.signals || {};
  row.appendChild(stateMeta([
    `signals ${Object.keys(signals).length}`,
    `variables ${Object.keys(variables).length}`,
    `rules ${(rulesData && rulesData.rules ? rulesData.rules.length : 0)}`
  ]));
  row.appendChild(jsonBlock({
    variables,
    posture_levels: state.posture_levels || {},
    active_rules: state.active_rules || [],
    last_signals: signals,
    recent_adjustments: (state.adjustments || []).slice(-5)
  }));
  const editor = document.createElement('textarea');
  editor.id = 'homeostasisRulesText';
  editor.rows = 8;
  editor.value = JSON.stringify((rulesData && rulesData.rules) || [], null, 2);
  row.appendChild(editor);
  const actions = el('div', 'state-actions');
  actions.appendChild(actionButton('Save Rules', saveHomeostasisRules));
  actions.appendChild(actionButton('Reset State', resetHomeostasis, 'danger ghost'));
  actions.appendChild(actionButton('Default Rules', resetHomeostasisRules));
  row.appendChild(actions);
  list.appendChild(row);
}

function renderEvolutionState(data) {
  const list = $('#evolutionList');
  list.innerHTML = '';
  if (!data) {
    list.appendChild(el('p', 'empty', 'Evolution status unavailable.'));
    return;
  }
  const head = el('div', 'state-row');
  head.appendChild(el('div', 'state-text', data.enabled ? 'Evolution proposals enabled.' : 'Evolution proposals disabled.'));
  head.appendChild(stateMeta([
    `${(data.proposals || []).length} proposal(s)`,
    `${(data.snapshots || []).length} snapshot(s)`,
    `canaries ${(data.canary_areas || []).length}`
  ]));
  const headActions = el('div', 'state-actions');
  headActions.appendChild(actionButton('Propose', proposeEvolution));
  headActions.appendChild(actionButton('Log Note', appendEvolutionLog));
  headActions.appendChild(actionButton('Rollback', rollbackEvolution, 'danger ghost'));
  head.appendChild(headActions);
  const logText = document.createElement('textarea');
  logText.id = 'evolutionLogText';
  logText.rows = 3;
  logText.placeholder = 'Performance note for future proposals';
  head.appendChild(logText);
  list.appendChild(head);

  const snapshots = data.snapshots || [];
  if (snapshots.length) {
    const snap = el('div', 'state-row');
    snap.appendChild(el('div', 'state-text', 'Snapshots'));
    snapshots.slice().reverse().slice(0, 5).forEach(item => {
      const line = el('div', 'state-actions');
      line.appendChild(el('span', 'memory-meta', `${item.id || ''} · ${(item.reason || '').slice(0, 80)}`));
      line.appendChild(actionButton('Restore', () => restoreEvolutionSnapshot(item.id), 'danger ghost'));
      snap.appendChild(line);
    });
    list.appendChild(snap);
  }

  const proposals = data.proposals || [];
  if (!proposals.length) {
    list.appendChild(el('p', 'empty', 'No proposals yet.'));
    return;
  }
  proposals.slice().reverse().slice(0, 8).forEach(proposal => {
    const row = el('div', 'state-row');
    row.appendChild(el('div', 'state-text', proposal.title || proposal.id || 'Evolution proposal'));
    row.appendChild(stateMeta([proposal.status || 'pending', proposal.id || '', (proposal.created_at || '').slice(0, 19)]));
    if (proposal.rationale) row.appendChild(el('div', 'state-text', proposal.rationale));
    // Stored proposals have no `preview` (only the immediate propose response
    // does), so show the mutation itself — approval must never be blind.
    row.appendChild(jsonBlock(proposal.preview || {
      target_failure: proposal.target_failure,
      expected_tradeoff: proposal.expected_tradeoff,
      prompt_overrides: proposal.prompt_overrides,
      hyperparameters: proposal.hyperparameters,
      attention_terms: proposal.attention_terms,
      canary_impact: proposal.canary_impact,
      risk_notes: proposal.risk_notes
    }));
    const actions = el('div', 'state-actions');
    if (proposal.status === 'pending') {
      actions.appendChild(actionButton('Approve', () => approveEvolution(proposal.id)));
      actions.appendChild(actionButton('Reject', () => rejectEvolution(proposal.id), 'danger ghost'));
    }
    row.appendChild(actions);
    list.appendChild(row);
  });
}

function renderContinuityState(selfModel, awarenessData, workspaceData, groundingData, presenceData) {
  const list = $('#continuityList');
  list.innerHTML = '';
  const row = el('div', 'state-row');
  row.appendChild(el('div', 'state-text', 'Continuity layers'));
  row.appendChild(stateMeta([
    `projects ${((selfModel || {}).current_projects || []).length}`,
    `promises ${((selfModel || {}).open_promises || []).length}`,
    `loops ${((awarenessData || {}).open_loops || []).length}`,
    `workspace ${(((workspaceData || {}).metrics || {}).content_count || 0)} items`,
    `integration proxy ${(((workspaceData || {}).metrics || {}).integration_proxy || 0)}`,
    `symbolic-only ${(((groundingData || {}).metrics || {}).symbolic_only || 0)}`,
    `presence evidence ${(presenceData || {}).turns_observed || 0} turns`,
    `phenomenal ${(presenceData || {}).phenomenal_status || (groundingData || {}).phenomenal_status || 'unknown'}`
  ]));
  row.appendChild(jsonBlock({
    self_model: selfModel || {},
    awareness: awarenessData || {},
    shared_workspace: workspaceData || {},
    semantic_grounding: groundingData || {},
    functional_social_presence: presenceData || {}
  }));
  const mood = el('div', 'state-form');
  [
    ['emotionHappiness', 'Happiness'],
    ['emotionStress', 'Stress'],
    ['emotionConfidence', 'Confidence'],
    ['emotionCuriosity', 'Curiosity']
  ].forEach(([id, label]) => {
    const input = document.createElement('input');
    input.id = id;
    input.type = 'number';
    input.min = '-1';
    input.max = '1';
    input.step = '0.05';
    input.placeholder = label;
    input.setAttribute('aria-label', label);
    mood.appendChild(input);
  });
  const reason = document.createElement('input');
  reason.id = 'emotionReason';
  reason.placeholder = 'Reason';
  reason.setAttribute('aria-label', 'Reason');
  mood.appendChild(reason);
  row.appendChild(mood);
  const actions = el('div', 'state-actions');
  actions.appendChild(actionButton('Set Mood', setEmotionManual));
  actions.appendChild(actionButton('Reset Mood', resetEmotion, 'danger ghost'));
  actions.appendChild(actionButton('Reset Awareness', resetAwareness, 'danger ghost'));
  actions.appendChild(actionButton('Reset Self Model', resetSelfModel, 'danger ghost'));
  actions.appendChild(actionButton('Reset Workspace', resetWorkspace, 'danger ghost'));
  actions.appendChild(actionButton('Reset Grounding', resetGrounding, 'danger ghost'));
  actions.appendChild(actionButton('Reset Presence Evidence', resetPresence, 'danger ghost'));
  row.appendChild(actions);
  list.appendChild(row);
}

async function loadSystems() {
  const [integrity, homeo, rules, evolution, selfModel, aware, workspace, grounding, presence] = await Promise.all([
    api('/api/integrity' + modelQuery()).catch(err => ({ ok: false, unavailable: true, summary: { errors: 1, warnings: 0 }, checks: { request: { count: 0, issues: [{ severity: 'error', code: 'request_failed', message: err.message }] } } })),
    api('/api/homeostasis').catch(() => null),
    api('/api/homeostasis/rules').catch(() => null),
    api('/api/evolution').catch(() => null),
    api('/api/self_model').catch(() => null),
    api('/api/awareness').catch(() => null),
    api('/api/workspace').catch(() => null),
    api('/api/grounding').catch(() => null),
    api('/api/presence').catch(() => null)
  ]);
  renderIntegrity(integrity);
  renderHomeostasisState(homeo, rules);
  renderEvolutionState(evolution);
  renderContinuityState(selfModel, aware, workspace, grounding, presence);
}

function openSystems() {
  systemsOpen = true;
  $('#systemsPanel').classList.remove('hidden');
  loadSystems().catch(err => { $('#integrityList').textContent = err.message; });
}

function closeSystems() {
  systemsOpen = false;
  $('#systemsPanel').classList.add('hidden');
}

async function resetHomeostasis() {
  if (!confirm('Reset homeostasis state?')) return;
  await api('/api/homeostasis/reset', { method: 'POST' });
  await loadSystems();
  addBubble('assistant', 'Homeostasis state reset.');
}

async function resetHomeostasisRules() {
  if (!confirm('Restore default homeostasis rules?')) return;
  await api('/api/homeostasis/rules', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rules: null })
  });
  await loadSystems();
  addBubble('assistant', 'Homeostasis rules restored to defaults.');
}

async function saveHomeostasisRules() {
  const area = $('#homeostasisRulesText');
  let rules;
  try {
    rules = JSON.parse(area.value || '[]');
  } catch (err) {
    addBubble('assistant', 'Homeostasis rules JSON is invalid: ' + err.message);
    return;
  }
  await api('/api/homeostasis/rules', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rules })
  });
  await loadSystems();
  addBubble('assistant', 'Homeostasis rules saved.');
}

async function resetAwareness() {
  if (!confirm('Reset awareness state?')) return;
  await api('/api/awareness/reset', { method: 'POST' });
  await loadSystems();
  loadAwareness().catch(() => {});
  addBubble('assistant', 'Awareness state reset.');
}

async function resetSelfModel() {
  if (!confirm('Reset self-model continuity state?')) return;
  await api('/api/self_model/reset', { method: 'POST' });
  await loadSystems();
  addBubble('assistant', 'Self model reset.');
}

async function resetWorkspace() {
  if (!confirm('Reset the recurrent workspace state?')) return;
  await api('/api/workspace/reset', { method: 'POST' });
  await loadSystems();
  addBubble('assistant', 'Shared workspace state reset.');
}

async function resetGrounding() {
  if (!confirm('Reset the semantic-grounding state?')) return;
  await api('/api/grounding/reset', { method: 'POST' });
  await loadSystems();
  addBubble('assistant', 'Semantic-grounding state reset.');
}

async function resetPresence() {
  if (!confirm('Reset the functional/social presence evidence ledger?')) return;
  await api('/api/presence/reset', { method: 'POST' });
  await loadSystems();
  addBubble('assistant', 'Presence evidence ledger reset. Phenomenal status remains unknown.');
}

async function setEmotionManual() {
  const payload = {};
  [
    ['emotionHappiness', 'happiness'],
    ['emotionStress', 'stress'],
    ['emotionConfidence', 'confidence'],
    ['emotionCuriosity', 'curiosity']
  ].forEach(([id, key]) => {
    const raw = $('#' + id).value.trim();
    if (!raw) return; // blank field = leave that dimension unchanged
    const value = Number(raw);
    if (Number.isFinite(value)) payload[key] = Math.max(-1, Math.min(1, value));
  });
  const reason = $('#emotionReason').value.trim();
  if (reason) payload.reason = reason;
  if (!Object.keys(payload).length) return;
  const state = await api('/api/emotion', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  await loadSystems();
  addBubble('assistant', `Mood updated: ${(state.labels || ['neutral']).join(', ')}`);
}

async function repairIntegrity() {
  const preview = await api('/api/integrity/repair' + modelQuery() + (modelQuery() ? '&' : '?') + 'dry_run=true', {
    method: 'POST'
  });
  if ((preview.repairable || 0) <= 0) {
    addBubble('assistant', 'No repairable integrity issues found.');
    return;
  }
  if (!confirm(`Repair ${preview.repairable} dangling graph edge(s)? This only deletes graph links whose backing memory is already gone.`)) return;
  const res = await api('/api/integrity/repair' + modelQuery() + (modelQuery() ? '&' : '?') + 'dry_run=false', {
    method: 'POST'
  });
  await loadSystems();
  addBubble('assistant', `Integrity repair complete: deleted ${res.deleted || 0} dangling graph edge(s).`);
}

async function proposeEvolution() {
  const res = await api('/api/evolution/propose', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: selectedModel() })
  });
  await loadSystems();
  addBubble('assistant', `Evolution proposal created:\n${res.title || res.id || 'proposal'}`);
}

async function appendEvolutionLog() {
  const entry = ($('#evolutionLogText') || {}).value || '';
  if (!entry.trim()) return;
  await api('/api/evolution/log', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entry: entry.trim() })
  });
  $('#evolutionLogText').value = '';
  await loadSystems();
  addBubble('assistant', 'Evolution note logged.');
}

async function approveEvolution(id) {
  if (!confirm('Approve this evolution proposal?')) return;
  const res = await api('/api/evolution/approve', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ proposal_id: id })
  });
  await loadSystems();
  loadRetrievalDefaults().catch(() => {});
  addBubble('assistant', `Approved evolution proposal ${res.proposal_id || id}.`);
}

async function rejectEvolution(id) {
  if (!confirm('Reject this evolution proposal?')) return;
  await api('/api/evolution/reject', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ proposal_id: id, reason: 'Rejected from Systems panel.' })
  });
  await loadSystems();
  addBubble('assistant', `Rejected evolution proposal ${id}.`);
}

async function restoreEvolutionSnapshot(id) {
  if (!confirm('Restore this evolution snapshot?')) return;
  const res = await api('/api/evolution/snapshots/restore', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ snapshot_id: id })
  });
  await loadSystems();
  loadRetrievalDefaults().catch(() => {});
  addBubble('assistant', `Restored evolution snapshot ${res.snapshot_id || id}.`);
}

async function rollbackEvolution() {
  if (!confirm('Rollback the active evolution configuration?')) return;
  const res = await api('/api/evolution/rollback', { method: 'POST' });
  await loadSystems();
  loadRetrievalDefaults().catch(() => {});
  addBubble('assistant', `Evolution rollback complete.\n${JSON.stringify(res, null, 2)}`);
}

function renderStateRows(id, rows, renderRow, emptyText) {
  const list = $(id);
  list.innerHTML = '';
  if (!rows || !rows.length) {
    list.appendChild(el('p', 'empty', emptyText));
    return;
  }
  rows.forEach(row => list.appendChild(renderRow(row)));
}

function renderCognitiveState(data) {
  renderStateRows('#goalList', data.goals, goal => {
    const row = el('div', 'state-row');
    row.appendChild(el('div', 'state-text', goal.text || ''));
    if (goal.next_action) row.appendChild(el('div', 'toolnote', goal.next_action));
    row.appendChild(stateMeta([goal.status, 'priority ' + (goal.priority || 3), (goal.due || '').slice(0, 10)]));
    const actions = el('div', 'state-actions');
    if (goal.status !== 'active') actions.appendChild(actionButton('Active', () => setGoalStatus(goal.id, 'active')));
    if (goal.status !== 'done') actions.appendChild(actionButton('Done', () => setGoalStatus(goal.id, 'done')));
    if (goal.status !== 'dropped') actions.appendChild(actionButton('Drop', () => setGoalStatus(goal.id, 'dropped'), 'danger ghost'));
    row.appendChild(actions);
    return row;
  }, 'No goals yet.');

  renderStateRows('#worldList', data.world, item => {
    const row = el('div', 'state-row');
    row.appendChild(el('div', 'state-text', `${item.key}: ${item.value}`));
    row.appendChild(stateMeta([item.status, `confidence ${item.confidence}`, item.expired ? 'expired' : '', item.source]));
    const actions = el('div', 'state-actions');
    if (item.status !== 'current') actions.appendChild(actionButton('Current', () => saveWorldItem({ ...item, status: 'current' })));
    if (item.status !== 'stale') actions.appendChild(actionButton('Stale', () => saveWorldItem({ ...item, status: 'stale' })));
    if (item.status !== 'archived') actions.appendChild(actionButton('Archive', () => saveWorldItem({ ...item, status: 'archived' }), 'danger ghost'));
    row.appendChild(actions);
    return row;
  }, 'No world facts yet.');

  renderStateRows('#beliefList', data.beliefs, belief => {
    const row = el('div', 'state-row');
    row.appendChild(el('div', 'state-text', belief.claim || ''));
    row.appendChild(stateMeta([belief.status, `confidence ${belief.confidence}`, belief.expired ? 'expired' : '', belief.source]));
    const actions = el('div', 'state-actions');
    if (belief.status !== 'active') actions.appendChild(actionButton('Active', () => saveBelief({ ...belief, status: 'active' })));
    if (belief.status !== 'stale') actions.appendChild(actionButton('Stale', () => saveBelief({ ...belief, status: 'stale' })));
    if (belief.status !== 'rejected') actions.appendChild(actionButton('Reject', () => saveBelief({ ...belief, status: 'rejected' }), 'danger ghost'));
    row.appendChild(actions);
    return row;
  }, 'No beliefs yet.');
}

async function loadCognitiveState() {
  const data = await api('/api/cognitive_state');
  renderCognitiveState(data);
}

function renderToolResults(results) {
  if (!results || !results.length) return;
  results.slice().reverse().forEach(call => {
    const result = call.result || {};
    const pending = result.pending && result.action;
    const title = pending
      ? `Pending ${call.tool}: ${result.action.id}`
      : `Tool ${call.tool}`;
    const row = el('div', 'card');
    row.appendChild(el('div', 'card-meta', title));
    row.appendChild(el('div', 'card-text', JSON.stringify(result, null, 2)));
    $('#recalled').prepend(row);
  });
  if (results.some(call => call.result && call.result.pending)) loadPendingActions().catch(() => {});
}

function setPendingActionCount(count) {
  pendingActionCount = Math.max(0, Number(count) || 0);
  const btn = $('#actionsBtn');
  if (!btn) return;
  btn.classList.toggle('pending-hot', pendingActionCount > 0);
  if (pendingActionCount > 0) {
    btn.dataset.count = String(pendingActionCount);
    btn.title = `${pendingActionCount} pending action${pendingActionCount === 1 ? '' : 's'}`;
  } else {
    delete btn.dataset.count;
    btn.title = '';
  }
}

function previewText(preview) {
  if (!preview || !Object.keys(preview).length) return '';
  if (preview.kind === 'file_write') {
    return [`${preview.mode || 'write'} ${preview.path || ''}`, `${preview.chars || 0} chars`, preview.diff || ''].filter(Boolean).join('\n');
  }
  if (preview.kind === 'source_of_truth_update') {
    return [`Update Source of Truth`, preview.path || '', `${preview.matches || 0} exact match(es)`, preview.diff || ''].filter(Boolean).join('\n');
  }
  if (preview.kind === 'email') {
    return [`To: ${preview.recipient || ''}`, `Subject: ${preview.subject || ''}`, '', preview.body || ''].join('\n');
  }
  if (preview.kind === 'action_package') {
    return [`Action Package: ${preview.title || ''}`, `External action authorized: ${preview.will_execute_external_action ? 'yes' : 'no'}`, '', preview.body || ''].join('\n');
  }
  if (preview.kind === 'google_calendar_create') {
    return [`Create Google Calendar event on ${preview.calendar_id || 'primary'}`, '', JSON.stringify(preview.event || {}, null, 2)].join('\n');
  }
  if (preview.kind === 'google_calendar_update') {
    return [`Update Google Calendar event: ${preview.current_title || preview.event_id || ''}`, `Calendar: ${preview.calendar_id || 'primary'}`, '', JSON.stringify(preview.changes || {}, null, 2)].join('\n');
  }
  if (preview.kind === 'google_calendar_delete') {
    return [`DELETE Google Calendar event: ${preview.current_title || preview.event_id || ''}`, `Calendar: ${preview.calendar_id || 'primary'}`, `Event ID: ${preview.event_id || ''}`].join('\n');
  }
  if (preview.kind === 'memory_merge') {
    return `Merge ${(preview.before || []).length} memories into:\n${(preview.after && preview.after.text) || ''}`;
  }
  if (preview.before || preview.after) {
    return JSON.stringify({ before: preview.before || null, after: preview.after || null }, null, 2);
  }
  return JSON.stringify(preview, null, 2);
}

function renderPendingActions(actions) {
  const list = $('#actionsList');
  list.innerHTML = '';
  setPendingActionCount(actions ? actions.length : 0);
  if (!actions || !actions.length) {
    list.appendChild(el('p', 'empty', 'No pending actions.'));
    return;
  }
  actions.forEach(action => {
    const row = el('div', 'action-row');
    const body = el('div');
    body.appendChild(el('div', 'action-title', action.tool || 'tool action'));
    body.appendChild(el('div', 'action-meta', `${action.policy || 'policy'} · ${action.id || ''} · ${(action.created_at || '').slice(0, 19)}`));
    const preview = previewText(action.preview);
    if (preview) body.appendChild(el('div', 'preview-block', preview));
    body.appendChild(el('div', 'action-args', JSON.stringify(action.args || {}, null, 2)));
    const buttons = el('div', 'action-buttons');
    buttons.appendChild(actionButton('Approve', () => approveAction(action.id)));
    buttons.appendChild(actionButton('Reject', () => rejectAction(action.id), 'danger ghost'));
    row.appendChild(body);
    row.appendChild(buttons);
    list.appendChild(row);
  });
}

async function loadPendingActions() {
  const data = await api('/api/tools/pending');
  const actions = data.actions || [];
  renderPendingActions(actions);
  return actions;
}

function openActions() {
  actionsOpen = true;
  $('#actionsPanel').classList.remove('hidden');
  loadPendingActions().catch(err => { $('#actionsList').textContent = err.message; });
}

function closeActions() {
  actionsOpen = false;
  $('#actionsPanel').classList.add('hidden');
}

async function approveAction(id) {
  const res = await api('/api/tools/pending/approve', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_id: id })
  });
  const tool = res.action.tool;
  const executionFailed = res.execution_failed || res.action.status === 'failed';
  const message = executionFailed
    ? `Approval accepted, but ${tool} failed.\n${JSON.stringify(res.result || {}, null, 2)}`
    : `Approved ${tool}.\n${JSON.stringify(res.result || {}, null, 2)}`;
  addBubble('assistant', message);
  rememberAssistantEvent(message);
  const remaining = await loadPendingActions();
  if (safetyOpen) loadSafety().catch(() => {});
  if (managerOpen) loadMemories().catch(() => {});
  loadStats().catch(() => {});
  // Approval is the user's answer to the model's gate, so resume the interrupted
  // turn automatically. Keep the synthetic instruction out of the visible
  // transcript; the approved result above and the eventual answer are the only
  // useful user-facing messages. If a turn queued several actions, wait until
  // all of them have been reviewed so the model does not resume prematurely.
  if (!remaining.length) {
    // A failed execution is already fully reported above. Do not feed it back
    // into generation: the model may "recover" by queuing the identical action
    // and create an approval loop. Let the user choose the next step.
    if (!executionFailed) {
      const continuation = [
          `The user approved ${tool} (${id}) and it has completed successfully.`,
          'Use the approved result already present in the conversation to resume and finish the original request.',
          'Do not ask for this approval again or rerun the completed action merely to verify it.'
        ].join(' ');
      await send(continuation, { showUser: false });
    }
  }
}

async function rejectAction(id) {
  const res = await api('/api/tools/pending/reject', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_id: id })
  });
  const message = `Rejected ${res.action.tool}.`;
  addBubble('assistant', message);
  rememberAssistantEvent(message);
  await loadPendingActions();
  if (safetyOpen) loadSafety().catch(() => {});
}

function renderPermissions(tools) {
  const list = $('#permissionList');
  list.innerHTML = '';
  if (!tools || !tools.length) {
    list.appendChild(el('p', 'empty', 'No tool permissions found.'));
    return;
  }
  tools.forEach(item => {
    const row = el('div', 'permission-row');
    const head = el('div', 'permission-head');
    const title = el('div');
    title.appendChild(el('div', 'permission-name', item.tool || 'tool'));
    title.appendChild(el('div', 'permission-meta', `${item.group || 'other'} · ${item.policy || 'policy'}`));
    const label = el('label', 'switch-row');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = !!item.enabled;
    checkbox.addEventListener('change', () => setToolPermission(item.tool, checkbox.checked));
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(item.enabled ? 'Enabled' : 'Disabled'));
    head.appendChild(title);
    head.appendChild(label);
    row.appendChild(head);
    list.appendChild(row);
  });
}

function renderAudit(events) {
  const list = $('#auditList');
  list.innerHTML = '';
  if (!events || !events.length) {
    list.appendChild(el('p', 'empty', 'No audited tool actions yet.'));
    return;
  }
  events.forEach(event => {
    const row = el('div', 'audit-row');
    const head = el('div', 'audit-head');
    const title = el('div');
    title.appendChild(el('div', 'audit-title', `${event.decision || 'event'} · ${event.tool || 'tool'}`));
    title.appendChild(el('div', 'audit-meta', `${event.policy || 'policy'} · ${event.action_id || ''} · ${(event.created_at || '').slice(0, 19)}`));
    head.appendChild(title);
    if (event.decision === 'approved' && event.action_id) {
      head.appendChild(actionButton('Undo', () => undoAction(event.action_id), 'ghost'));
    }
    row.appendChild(head);
    row.appendChild(el('div', 'audit-body', JSON.stringify({ args: event.args || {}, result: event.result || null }, null, 2)));
    list.appendChild(row);
  });
}

async function loadSafety() {
  const permissions = await api('/api/tools/permissions');
  const audit = await api('/api/tools/audit?limit=100');
  renderPermissions(permissions.tools || []);
  renderAudit(audit.events || []);
}

function openSafety() {
  safetyOpen = true;
  $('#safetyPanel').classList.remove('hidden');
  loadSafety().catch(err => { $('#auditList').textContent = err.message; });
}

function closeSafety() {
  safetyOpen = false;
  $('#safetyPanel').classList.add('hidden');
}

async function setToolPermission(tool, enabled) {
  await api('/api/tools/permissions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool, enabled })
  });
  await loadSafety();
}

async function undoAction(id) {
  const res = await api('/api/tools/pending/undo', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_id: id })
  });
  addBubble('assistant', `Undid ${res.action.tool}.\n${JSON.stringify(res.result || {}, null, 2)}`);
  await loadSafety();
  if (actionsOpen) loadPendingActions().catch(() => {});
  if (managerOpen) loadMemories().catch(() => {});
  loadStats().catch(() => {});
}

function openState() {
  stateOpen = true;
  $('#statePanel').classList.remove('hidden');
  loadCognitiveState().catch(err => { $('#goalList').textContent = err.message; });
}

function closeState() {
  stateOpen = false;
  $('#statePanel').classList.add('hidden');
}

async function setGoalStatus(id, status) {
  await api('/api/cognitive_state/goal/' + encodeURIComponent(id) + '/status', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status })
  });
  await loadCognitiveState();
}

async function saveWorldItem(item) {
  await api('/api/cognitive_state/world', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: item.id,
      key: item.key,
      value: item.value,
      confidence: Number(item.confidence ?? 0.7),
      source: item.source || null,
      expires: item.expires || null,
      status: item.status || 'current'
    })
  });
  await loadCognitiveState();
}

async function saveBelief(belief) {
  await api('/api/cognitive_state/belief', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: belief.id,
      claim: belief.claim,
      confidence: Number(belief.confidence ?? 0.5),
      source: belief.source || null,
      expires: belief.expires || null,
      last_verified: belief.last_verified || null,
      status: belief.status || 'active'
    })
  });
  await loadCognitiveState();
}

function shortSource(source) {
  if (source && source.startsWith('upload:')) return source.split(':').pop();
  return source ? source.split('/').pop() : '';
}

function sourceLabel(source) {
  if (!source) return '';
  if (source.startsWith('upload:')) {
    const parts = source.split(':');
    const hash = parts[1] || '';
    const filename = parts.slice(2).join(':') || source;
    return hash ? `${filename} · ${hash.slice(0, 8)}` : filename;
  }
  return source.split('/').pop();
}

function populateSelect(sel, values, label) {
  const current = sel.value;
  sel.innerHTML = '';
  const all = el('option', null, label); all.value = ''; sel.appendChild(all);
  values.forEach(v => { const o = el('option', null, sourceLabel(v) || v); o.value = v; sel.appendChild(o); });
  if ([...sel.options].some(o => o.value === current)) sel.value = current;
}

async function loadMemories() {
  const params = new URLSearchParams();
  const q = $('#memSearch').value.trim();
  if (q) params.set('q', q);
  if ($('#memCategory').value) params.set('category', $('#memCategory').value);
  if ($('#memSource').value) params.set('source', $('#memSource').value);
  if (selectedModel()) params.set('model', selectedModel());
  const data = await api('/api/memories?' + params.toString());
  setStats(data.count);
  populateSelect($('#memCategory'), data.categories || [], 'All categories');
  populateSelect($('#memSource'), data.sources || [], 'All sources');
  renderMemoryList(data.memories || []);
}

function renderMemoryList(memories) {
  const list = $('#memoryList');
  list.innerHTML = '';
  if (!memories.length) { list.appendChild(el('p', 'empty', 'No memories match the current filters.')); return; }
  memories.forEach(m => {
    const row = el('div', 'memory-row');
    const body = el('div');
    body.appendChild(el('div', 'memory-text', m.text));
    if (m.pinned) body.appendChild(el('div', 'toolnote', 'Pinned'));
    body.appendChild(el('div', 'memory-meta',
      `${m.category || 'uncategorized'} · ${m.source_type || 'legacy'} · importance ${m.importance || 3} · ${(m.date || '').slice(0, 10)} · ${m.mode}${m.source ? ' · ' + sourceLabel(m.source) : ''}`));
    const actions = el('div', 'memory-actions');
    const detail = el('button', 'ghost', 'Detail');
    detail.type = 'button';
    detail.addEventListener('click', () => showMemoryDetail(m.id));
    actions.appendChild(detail);
    const recall = el('button', 'ghost', 'Recall');
    recall.type = 'button';
    recall.addEventListener('click', () => activateMemory(m));
    actions.appendChild(recall);
    const pin = el('button', 'ghost', m.pinned ? 'Unpin' : 'Pin');
    pin.type = 'button';
    pin.addEventListener('click', () => pinMemory(m.id, !m.pinned));
    actions.appendChild(pin);
    const del = el('button', 'danger', 'Delete');
    del.type = 'button';
    del.addEventListener('click', () => deleteMemory(m.id));
    actions.appendChild(del);
    if (m.source) {
      const delSource = el('button', 'danger ghost', 'Delete file');
      delSource.type = 'button';
      delSource.title = m.source;
      delSource.addEventListener('click', () => deleteSource(m.source));
      actions.appendChild(delSource);
    }
    row.appendChild(body);
    row.appendChild(actions);
    list.appendChild(row);
  });
}

function kv(label, value) {
  const row = el('div', 'kv');
  row.appendChild(el('div', 'k', label));
  row.appendChild(el('div', null, value == null || value === '' ? '—' : String(value)));
  return row;
}

async function showMemoryDetail(id) {
  try {
    const params = selectedModel() ? '?model=' + encodeURIComponent(selectedModel()) : '';
    const m = await api('/api/memory/' + encodeURIComponent(id) + params);
    $('#detailTitle').textContent = m.category || 'Memory';
    const body = $('#detailBody');
    body.innerHTML = '';
    body.appendChild(kv('ID', m.id));
    body.appendChild(kv('Category', m.category));
    body.appendChild(kv('Source type', m.source_type));
    body.appendChild(kv('Source', sourceLabel(m.source) || m.source));
    body.appendChild(kv('Store', m.mode));
    body.appendChild(kv('Pinned', m.pinned ? 'yes' : 'no'));
    body.appendChild(el('div', 'memory-text', m.text || ''));
    body.appendChild(el('div', 'mono', JSON.stringify(m.metadata || {}, null, 2)));
    $('#detailPanel').classList.remove('hidden');
  } catch (e) {
    addBubble('assistant', 'Detail failed: ' + e.message);
  }
}

async function pinMemory(id, pinned) {
  await api('/api/memory/' + encodeURIComponent(id) + '/pin', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pinned, model: selectedModel() })
  });
  if (managerOpen) await loadMemories();
}

async function deleteMemory(id) {
  if (!confirm('Delete this memory? A JSON backup will be created first.')) return;
  await api('/api/memory/' + encodeURIComponent(id) + modelQuery(), { method: 'DELETE' });
  await loadMemories(); await loadStats();
}

async function deleteSource(source) {
  const preview = await api('/api/memory/delete_source/preview', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, model: selectedModel() })
  });
  if (!confirm(`Delete ${preview.count} memory chunk(s) from this source? A JSON backup will be created first.`)) return;
  const res = await api('/api/memory/delete_source', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, model: selectedModel() })
  });
  setStats(res.count);
  if (res.backup) addBubble('assistant', 'Backup created before source deletion:\n' + res.backup);
  await loadMemories();
}

function exportFiltered() {
  const params = new URLSearchParams();
  params.set('format', 'json');
  if (selectedModel()) params.set('model', selectedModel());
  if ($('#memCategory').value) params.set('category', $('#memCategory').value);
  if ($('#memSource').value) params.set('source', $('#memSource').value);
  window.location.href = '/api/export?' + params.toString();
}

async function runRecallQuery() {
  const query = $('#recallQuery').value.trim();
  if (!query) return;
  const res = await api('/api/recall', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, retrieval: retrievalSettings(), model: selectedModel() })
  });
  renderRecalled(res.memories || []);
  addBubble('assistant', `Recall query returned ${(res.memories || []).length} memor${(res.memories || []).length === 1 ? 'y' : 'ies'}.`);
}

async function summarizeSelectedCategory() {
  let category = $('#memCategory').value || $('#wCategory').value || '';
  if (!category) category = prompt('Category to summarize') || '';
  category = category.trim();
  if (!category) return;
  if (!confirm(`Summarize category "${category}"? An LLM pass will write a summary; nothing is deleted in this step.`)) return;
  const replace = confirm(`Also REPLACE the original "${category}" memories with the summary?\nOK = replace (a backup is made first) · Cancel = keep originals.`);
  const res = await api('/api/summarize', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ category, replace, model: selectedModel() })
  });
  addBubble('assistant', res.summary || `No memories found in category "${category}".`);
  if (res.backup) addBubble('assistant', 'Backup created before summarize replace:\n' + res.backup);
  await loadStats().catch(() => {});
  if (managerOpen) loadMemories().catch(() => {});
}

function openManager() {
  managerOpen = true;
  $('#manager').classList.remove('hidden');
  loadMemories().catch(err => { $('#memoryList').textContent = err.message; });
}

function closeManager() {
  managerOpen = false;
  $('#manager').classList.add('hidden');
}

let chatAbort = null;
let chatReqId = null;

function markStopped(bubble) {
  setBubbleText(bubble, 'Stopped.');
  // Nothing to roll back: a turn only enters history as a complete pair once
  // its reply arrives (see commitChatTurn), so a stopped request left no trace.
}

function commitChatTurn(message, reply) {
  // The user+assistant pair enters history ATOMICALLY, only when the reply
  // arrives. Pushing the user turn at send time and the reply at completion
  // time corrupted history under concurrent sends: two pending user rows sat
  // adjacent (cleanHistory collapses same-role neighbors, keeping the later),
  // and completion order decided which reply landed next to which question.
  history = cleanHistory(history);
  history.push({ role: 'user', content: message });
  history.push({ role: 'assistant', content: reply });
  history = cleanHistory(history).slice(-HISTORY_LIMIT);
  saveHistory();
}

function cleanHistory(items) {
  const cleaned = [];
  for (const item of items || []) {
    if (!item || (item.role !== 'user' && item.role !== 'assistant')) continue;
    const content = String(item.content || '').trim();
    if (!content) continue;
    // A runaway model response must never be persisted or sent back in the
    // next request. The transcript may still display it for diagnosis.
    if (content.length > MAX_HISTORY_MESSAGE_CHARS) continue;
    const msg = { role: item.role, content };
    if (cleaned.length && cleaned[cleaned.length - 1].role === msg.role) {
      cleaned[cleaned.length - 1] = msg;
    } else {
      cleaned.push(msg);
    }
  }
  while (cleaned.length && cleaned[0].role !== 'user') cleaned.shift();
  // History is prior context, not an in-flight turn. Never send an unmatched
  // user message: the current message has its own top-level API field.
  if (cleaned.length && cleaned[cleaned.length - 1].role !== 'assistant') cleaned.pop();
  return cleaned;
}

function chatHistoryForRequest() {
  return cleanHistory(history).slice(-HISTORY_LIMIT);
}

function restoreChatState() {
  try {
    const savedTranscript = JSON.parse(localStorage.getItem(CHAT_TRANSCRIPT_KEY) || '[]');
    chatTranscript = Array.isArray(savedTranscript)
      ? savedTranscript
          .filter(item => item && (item.role === 'user' || item.role === 'assistant') && item.content != null)
          .map((item, index) => ({ id: Number(item.id) || index + 1, role: item.role, content: String(item.content) }))
          .slice(-CHAT_TRANSCRIPT_LIMIT)
      : [];
  } catch (e) {
    chatTranscript = [];
  }
  nextTranscriptId = Math.max(1, ...chatTranscript.map(item => item.id || 0)) + 1;
  for (const item of chatTranscript) addBubble(item.role, item.content, { persist: false });

  try {
    history = cleanHistory(JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || '[]')).slice(-HISTORY_LIMIT);
  } catch (e) {
    history = [];
  }
  try {
    const savedActive = JSON.parse(localStorage.getItem(ACTIVE_MEMORY_KEY) || '[]');
    activeMemories = Array.isArray(savedActive)
      ? savedActive
          .filter(item => item && item.id && item.text != null)
          .map(item => ({
            id: String(item.id),
            text: String(item.text || ''),
            category: String(item.category || 'memory'),
            date: String(item.date || ''),
            source: item.source || null,
            pinned: !!item.pinned
          }))
          .slice(0, 10)
      : [];
  } catch (e) {
    activeMemories = [];
  }
  renderActiveMemories();
}

function clearChatState() {
  history = [];
  chatTranscript = [];
  activeMemories = [];
  nextTranscriptId = 1;
  try {
    localStorage.removeItem(CHAT_HISTORY_KEY);
    localStorage.removeItem(CHAT_TRANSCRIPT_KEY);
    localStorage.removeItem(ACTIVE_MEMORY_KEY);
  } catch (e) {}
  $('#messages').innerHTML = '';
  renderActiveMemories();
  clearContextUsage();
}

async function handleLocalCommand(text) {
  const parts = text.trim().split(/\s+/);
  const cmd = (parts[0] || '').toLowerCase();
  if (!cmd.startsWith('/')) return false;

  if (cmd === '/clear') {
    const removed = history.length;
    clearChatState();
    addBubble('assistant', `Chat context cleared. Removed ${removed} stored message${removed === 1 ? '' : 's'}, activated memories, and the displayed context measurement. Long-term Neural Map memory was preserved; use /purge to delete it.`);
    return true;
  }

  if (cmd === '/purge') {
    const removed = history.length;
    sessionMemoryEnabled = false;
    sessionStorage.setItem(SESSION_MEMORY_KEY, '0');
    clearChatState();
    renderRecalled([]);
    lastTurnObservability = null;
    addBubble('assistant', `Fresh chat started. Removed ${removed} chat messages and disconnected this session from long-term retrieval and persistent model-state prompts. Stored Neural Map memories were not deleted. Use /memory on to reconnect them.`);
    return true;
  }

  if (cmd === '/memory') {
    const setting = (parts[1] || '').toLowerCase();
    if (setting === 'on') {
      sessionMemoryEnabled = true;
      sessionStorage.setItem(SESSION_MEMORY_KEY, '1');
      addBubble('assistant', 'Neural Map memory reconnected for this session.');
    } else if (setting === 'off') {
      sessionMemoryEnabled = false;
      sessionStorage.setItem(SESSION_MEMORY_KEY, '0');
      addBubble('assistant', 'Neural Map memory disconnected for this session.');
    } else {
      addBubble('assistant', `Session memory is ${sessionMemoryEnabled ? 'on' : 'off'}. Use /memory on or /memory off.`);
    }
    return true;
  }

  if (cmd === '/trim') {
    const requested = Number.parseInt(parts[1] || '', 10);
    const keep = Number.isFinite(requested) ? Math.max(0, Math.min(requested, HISTORY_LIMIT)) : 4;
    const before = history.length;
    history = keep > 0 ? cleanHistory(history).slice(-keep) : [];
    history = cleanHistory(history);
    saveHistory();
    const removed = Math.max(0, before - history.length);
    addBubble('assistant', `Chat context trimmed to ${history.length} message${history.length === 1 ? '' : 's'} (${removed} removed).`);
    return true;
  }

  if (cmd === '/context') {
    history = cleanHistory(history);
    saveHistory();
    addBubble('assistant', `Short-term chat context: ${history.length} stored message${history.length === 1 ? '' : 's'} and ${activeMemories.length} activated memor${activeMemories.length === 1 ? 'y' : 'ies'}; session memory is ${sessionMemoryEnabled ? 'on' : 'off'}. ${Math.min(history.length, HISTORY_LIMIT)} message${Math.min(history.length, HISTORY_LIMIT) === 1 ? '' : 's'} will be sent with the next request.`);
    return true;
  }

  if (cmd === '/help') {
    addBubble('assistant', 'Commands: /search <topic> runs a general public-web search; /context shows chat-history size and memory mode; /trim keeps recent messages; /clear clears local chat history; /purge starts a fresh memory-disconnected chat; /memory on reconnects Neural Map memory.');
    return true;
  }

  return false;
}

async function send(message, options = {}) {
  if (options.showUser !== false) addBubble('user', message);
  // Build this request's context WITHOUT mutating shared history: the pair is
  // committed only on completion (commitChatTurn), so overlapping sends can
  // never interleave half-turns.
  // ``message`` is sent separately below; duplicating it in history used one
  // protocol slot and guillotined the oldest completed turn at a full window.
  const requestHistory = chatHistoryForRequest();
  const bubble = addBubble('assistant', '…', { persist: false }); bubble.classList.add('typing');
  const ctrl = new AbortController();
  const reqId = (crypto.randomUUID && crypto.randomUUID()) || (Date.now() + '-' + Math.random().toString(36).slice(2));
  chatAbort = ctrl;   // Stop always targets the most recent request
  chatReqId = reqId;
  $('#stopBtn').disabled = false;
  try {
	    const res = await api('/api/chat', {
	      method: 'POST', headers: { 'Content-Type': 'application/json' },
	      signal: ctrl.signal,
	      body: JSON.stringify({
          message,
          model: selectedModel(),
          num_ctx: parseInt($('#ctx').value) || null,
          history: requestHistory,
          activated_memory_ids: activeMemories.map(m => m.id),
          retrieval: retrievalSettings(),
          initiative_mode: selectedInitiativeMode(),
          memory_enabled: sessionMemoryEnabled,
          request_id: reqId
        })
	    });
    if (res.cancelled) {   // server confirmed the generation was aborted
      bubble.classList.remove('typing');
      markStopped(bubble);
      return;
	    }
	    bubble.classList.remove('typing');
	    if (res.context_usage) {
	      accumulateContextSession(res.context_usage);
	      renderContextUsage(res.context_usage);
	    }
      else if (!res.pending_decision) markContextUsageUnavailable();
	    setBubbleText(bubble, res.reply);
      recordBubble(bubble, 'assistant', res.reply);
      commitChatTurn(message, res.reply);
    renderRecalled(res.memories);
    if (res.reminders) renderReminders(res.reminders);
    if (res.awareness) renderAwareness(res.awareness, res.attention || null);
    if (res.initiatives) {
      renderInitiatives(res.initiatives);
      if (res.initiatives.length) {
        api('/api/initiative/contact', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ids: res.initiatives.map(i => i.id), mode: selectedInitiativeMode(), model: selectedModel() })
        }).catch(() => {});
      }
    }
    const toolResults = res.tool_results || [];
    const pendingTools = toolResults.filter(call => call.result && call.result.pending).map(call => call.tool);
    const usedTools = (res.tools_used || []).filter(name => !pendingTools.includes(name));
    if (pendingTools.length) {
      setPendingActionCount(Math.max(pendingActionCount, pendingTools.length));
      $('#recalled').prepend(el('p', 'toolnote', 'Pending approval: ' + pendingTools.join(', ')));
    }
    if (usedTools.length) {
      $('#recalled').prepend(el('p', 'toolnote', 'Used: ' + usedTools.join(', ')));
    }
    if (res.memory_enabled === false) {
      $('#recalled').prepend(el('p', 'toolnote', 'Fresh chat: long-term memory and persistent model state are disconnected.'));
    }
    renderToolResults(toolResults);
    lastTurnObservability = collectTurnObservability(res, toolResults, usedTools, pendingTools);
    if (observabilityOpen) loadObservability().catch(() => {});
	    if (res.thinking && res.thinking.path) {
	      const label = res.thinking.path === 'system2' ? 'System 2' : res.thinking.path === 'system1' ? 'System 1' : 'Standard';
	      $('#recalled').prepend(el('p', 'toolnote', `${label}: ${res.thinking.reason || 'decision gate'}`));
	    }
	    if (res.trace && res.trace.id) {
	      $('#recalled').prepend(el('p', 'toolnote', `Trace ${res.trace.id}: ${formatTraceMetrics(res.trace)}`));
	    }
	    if (res.council && res.council.used) {
	      $('#recalled').prepend(el('p', 'toolnote', `Inner Council: ${res.council.personas.join(', ')} · ${res.council.turns} rounds`));
	    }
    if (res.capture_error) $('#recalled').prepend(el('p', 'empty', res.capture_error));
    if (res.web_data && res.web_data.errors && res.web_data.errors.length) {
      $('#recalled').prepend(el('p', 'empty', 'Web search warning: ' + res.web_data.errors.map(e => e.error).join('; ')));
    }
    if (res.web_data && res.web_data.results && res.web_data.results.length) {
      $('#recalled').prepend(el('p', 'empty', `Web search: ${res.web_data.results.length} result(s) provided.`));
    }
    const webTool = toolResults.find(call => call.tool === 'web_search' && call.result);
    if (webTool && webTool.result) {
      const webResult = webTool.result;
      if (webResult.error) {
        $('#recalled').prepend(el('p', 'empty', `Web search failed: ${webResult.error}`));
      } else {
        const count = (webResult.results || []).length;
        const provider = webResult.provider || 'provider';
        const warnings = (webResult.errors || []).map(e => e.error || JSON.stringify(e)).filter(Boolean);
        $('#recalled').prepend(el('p', count ? 'toolnote' : 'empty', `Web search: ${count} result(s) from ${provider}.`));
        if (warnings.length) $('#recalled').prepend(el('p', 'empty', 'Web search warning: ' + warnings.join('; ')));
      }
    }
    if (res.count != null) setStats(res.count);
    if (res.pending_decision) loadPendingActions().catch(() => {});
  } catch (e) {
    bubble.classList.remove('typing');
    if (e.name === 'AbortError') {
      // The cancel POST (sent by the Stop handler) makes the server close its
      // Ollama stream, so the generation itself is aborted, not just the wait.
      markStopped(bubble);
	    } else {
	      setBubbleText(bubble, 'Error: ' + e.message);
        recordBubble(bubble, 'assistant', 'Error: ' + e.message);
	    }
  } finally {
    if (chatAbort === ctrl) {   // don't clobber a newer in-flight request's controller
      chatAbort = null;
      chatReqId = null;
      $('#stopBtn').disabled = true;
    }
  }
  $('#messages').scrollTop = $('#messages').scrollHeight;
}

function autosize() { const i = $('#input'); i.style.height = 'auto'; i.style.height = Math.min(i.scrollHeight, 160) + 'px'; }

async function fileAsBase64(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  const size = 0x8000;
  for (let i = 0; i < bytes.length; i += size) {
    binary += String.fromCharCode(...bytes.subarray(i, i + size));
  }
  return btoa(binary);
}

$('#composer').addEventListener('submit', async e => {
  e.preventDefault();
  const t = $('#input').value.trim(); if (!t) return;
  $('#input').value = ''; autosize();
  if (await handleLocalCommand(t)) return;
  send(t);
});
$('#input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); }
});
$('#stopBtn').addEventListener('click', () => {
  // Tell the server first so it aborts the Ollama generation, then drop the
  // local fetch. Fire-and-forget: the AbortError path renders the stopped state.
  if (chatReqId) {
    api('/api/chat/cancel', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: chatReqId, allow_pending: true })
    }).catch(() => {});
  }
  if (chatAbort) chatAbort.abort();
});
$('#input').addEventListener('input', autosize);
$('#model').addEventListener('change', onModelChange);
$('#ctx').addEventListener('change', () => {
  localStorage.setItem('nm_ctx', $('#ctx').value);
  clearContextUsage();
});
$('#initiativeMode').addEventListener('change', () => {
  localStorage.setItem('nm_initiative_mode', selectedInitiativeMode());
  loadInitiatives();
  loadAwareness();
});
$('#briefingBtn').addEventListener('click', showBriefing);
$('#willBtn').addEventListener('click', runWillReview);
$('#ingestBtn').addEventListener('click', () => $('#ingestFile').click());
$('#ingestFile').addEventListener('change', async e => {
  const files = [...e.target.files];
  if (!files.length) return;
  const btn = $('#ingestBtn'), orig = btn.textContent;
  let chunks = 0;
  for (const f of files) {
    btn.textContent = 'ingesting ' + f.name + '…';
    try {
      // PDFs and images are binary: send raw bytes (base64) for server-side
      // extraction / vision captioning. Text documents send their text directly.
      const isBinary = /\.(pdf|png|jpe?g|gif|webp|bmp|tiff?|wav|mp3|m4a|aac|flac|ogg|aiff?)$/i.test(f.name);
      const body = isBinary
        ? { filename: f.name, data_base64: await fileAsBase64(f), model: selectedModel() }
        : { filename: f.name, text: await f.text(), model: selectedModel() };
      const preview = await api('/api/ingest/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (preview.replace_count > 0 && !confirm(`${f.name} will replace ${preview.replace_count} older chunk(s). Continue? A JSON backup will be created first.`)) {
        continue;
      }
      const res = await api('/api/ingest', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      chunks += res.chunks; setStats(res.count); if (managerOpen) loadMemories();
    } catch (err) {
      const warn = el('p', 'empty', f.name + ': ' + err.message);
      $('#recalled').prepend(warn);
    }
  }
  e.target.value = '';
  btn.textContent = '✓ ' + chunks + ' chunks';
  setTimeout(() => { btn.textContent = orig; }, 2500);
});
$('#toggleAdd').addEventListener('click', () => $('#addForm').classList.toggle('hidden'));
$('#addForm').addEventListener('submit', async e => {
  e.preventDefault();
  const text = $('#memText').value.trim(); if (!text) return;
  try {
	    const res = await api('/api/memory', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, category: $('#memCat').value || 'note', importance: parseInt($('#memImp').value) || 3, model: selectedModel() })
    });
	    $('#memText').value = '';
	    setStats(res.count);
	    if (managerOpen) loadMemories();
	    $('#addForm').classList.add('hidden');
  } catch (err) {
    $('#memText').value = err.message;
  }
});

$('#manageBtn').addEventListener('click', openManager);
$('#manageTopBtn').addEventListener('click', openManager);
$('#stateBtn').addEventListener('click', openState);
$('#systemsBtn').addEventListener('click', openSystems);
$('#observabilityBtn').addEventListener('click', openObservability);
$('#actionsBtn').addEventListener('click', openActions);
$('#safetyBtn').addEventListener('click', openSafety);
$('#retrievalBtn').addEventListener('click', () => $('#retrievalPanel').classList.toggle('hidden'));
$('#reflectBtn').addEventListener('click', async () => {
  const btn = $('#reflectBtn'); const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Reflecting…';
  try {
    const res = await api('/api/reflect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: selectedModel() }) });
    renderReminders(res.active_reminders);
    renderInitiatives(res.active_initiatives);
    if (res.count != null) setStats(res.count);
    addBubble('assistant', `🧠 Reflection complete — ${res.insights} insight(s), ${res.reminders} reminder(s), ${res.initiatives || 0} initiative(s).`);
  } catch (e) { addBubble('assistant', 'Reflect failed: ' + e.message); }
  btn.disabled = false; btn.textContent = label;
});
$('#compactBtn').addEventListener('click', async () => {
  let preview;
  try {
    preview = await api('/api/compact', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: selectedModel(), dry_run: true }) });
  } catch (e) { addBubble('assistant', 'Compact preview failed: ' + e.message); return; }
  if (preview.eligible <= 0) { addBubble('assistant', 'No old unpinned episodes are eligible for compaction.'); return; }
  if (!confirm(`Compact ${preview.eligible} old episode(s) into one digest memory? A JSON backup will be created first.`)) return;
  const btn = $('#compactBtn'); const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Compacting…';
  try {
    const res = await api('/api/compact', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: selectedModel() }) });
    if (res.count != null) setStats(res.count);
    addBubble('assistant', `Compacted ${res.compacted} old episode(s) into a digest.${res.backup ? '\nBackup: ' + res.backup : ''}`);
    if (managerOpen) loadMemories();
  } catch (e) { addBubble('assistant', 'Compact failed: ' + e.message); }
  btn.disabled = false; btn.textContent = label;
});
$('#sleepBtn').addEventListener('click', async () => {
  let preview;
  try {
    preview = await api('/api/sleep', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: selectedModel(), dry_run: true }) });
  } catch (e) { addBubble('assistant', 'Sleep preview failed: ' + e.message); return; }
  if ((preview.archivable || 0) <= 0 && (preview.dreamable || 0) <= 0) {
    addBubble('assistant', 'Sleep cycle found nothing to archive or dream about.');
    return;
  }
  if (!confirm(`Run sleep cycle?\n\nArchive ${preview.archivable || 0} weak episode(s) to cold storage and create up to ${preview.dreamable || 0} speculative dream(s). A JSON backup will be created first.`)) return;
  const btn = $('#sleepBtn'); const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Sleeping…';
  try {
    const res = await api('/api/sleep', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: selectedModel() }) });
    if (res.count != null) setStats(res.count);
    addBubble('assistant', `Sleep cycle complete: archived ${res.archived || 0} episode(s), created ${res.dreamed || 0} speculative dream(s).${res.archive_path ? '\nCold storage: ' + res.archive_path : ''}${res.backup ? '\nBackup: ' + res.backup : ''}`);
    if (managerOpen) loadMemories();
  } catch (e) { addBubble('assistant', 'Sleep cycle failed: ' + e.message); }
  btn.disabled = false; btn.textContent = label;
});
$('#pruneBtn').addEventListener('click', async () => {
  let preview;
  try {
    preview = await api('/api/prune', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: selectedModel(), dry_run: true }) });
  } catch (e) { addBubble('assistant', 'Prune preview failed: ' + e.message); return; }
  if (preview.pruned <= 0) { addBubble('assistant', 'No faded episodes are eligible for pruning.'); return; }
  if (!confirm(`Forget ${preview.pruned} old, low-importance, never-recalled episode(s)? Facts, documents, and pinned memories are kept. A JSON backup will be created first.`)) return;
  const btn = $('#pruneBtn'); const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Pruning…';
  try {
    const res = await api('/api/prune', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: selectedModel() }) });
    if (res.count != null) setStats(res.count);
    addBubble('assistant', `Pruned ${res.pruned} faded episode(s).${res.backup ? '\nBackup: ' + res.backup : ''}`);
  } catch (e) { addBubble('assistant', 'Prune failed: ' + e.message); }
  btn.disabled = false; btn.textContent = label;
});
$('#resetRetrieval').addEventListener('click', () => setRetrieval(retrievalDefaults));
$('#recallBtn').addEventListener('click', () => runRecallQuery().catch(err => addBubble('assistant', 'Recall failed: ' + err.message)));
$('#recallQuery').addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  e.preventDefault();
  $('#recallBtn').click();
});
Object.values(retrievalInputs()).forEach(input => input.addEventListener('input', syncRetrievalLabels));
$('#closeManager').addEventListener('click', closeManager);
$('#closeState').addEventListener('click', closeState);
$('#closeSystems').addEventListener('click', closeSystems);
$('#closeObservability').addEventListener('click', closeObservability);
$('#closeActions').addEventListener('click', closeActions);
$('#closeSafety').addEventListener('click', closeSafety);
$('#refreshSystems').addEventListener('click', () => loadSystems());
$('#refreshObservability').addEventListener('click', () => loadObservability());
$('#refreshActions').addEventListener('click', () => loadPendingActions());
$('#refreshSafety').addEventListener('click', () => loadSafety());
$('#refreshMemories').addEventListener('click', () => loadMemories());
$('#summarizeCategory').addEventListener('click', () => summarizeSelectedCategory().catch(err => addBubble('assistant', 'Summarize failed: ' + err.message)));
$('#exportFiltered').addEventListener('click', exportFiltered);
$('#diagnosticsBtn').addEventListener('click', showDiagnostics);
$('#resetEmotionBtn').addEventListener('click', resetEmotion);
$('#closeDetail').addEventListener('click', () => $('#detailPanel').classList.add('hidden'));
$('#memSearch').addEventListener('input', () => loadMemories());
$('#memCategory').addEventListener('change', () => loadMemories());
$('#memSource').addEventListener('change', () => loadMemories());
$('#goalForm').addEventListener('submit', async e => {
  e.preventDefault();
  const text = $('#goalText').value.trim();
  if (!text) return;
  try {
    await api('/api/cognitive_state/goal', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        status: $('#goalStatus').value,
        priority: parseInt($('#goalPriority').value) || 3,
        evidence: splitStateList($('#goalEvidence').value),
        next_action: $('#goalNext').value.trim() || null
      })
    });
    $('#goalText').value = ''; $('#goalNext').value = ''; $('#goalEvidence').value = '';
    await loadCognitiveState();
  } catch (err) {
    $('#goalList').prepend(el('p', 'empty', err.message));
  }
});
$('#worldForm').addEventListener('submit', async e => {
  e.preventDefault();
  const key = $('#worldKey').value.trim();
  const value = $('#worldValue').value.trim();
  if (!key || !value) return;
  try {
    await saveWorldItem({
      key,
      value,
      status: $('#worldStatus').value,
      confidence: confidenceValue('#worldConfidence', 0.7),
      source: $('#worldSource').value.trim() || null
    });
    $('#worldKey').value = ''; $('#worldValue').value = ''; $('#worldSource').value = '';
  } catch (err) {
    $('#worldList').prepend(el('p', 'empty', err.message));
  }
});
$('#beliefForm').addEventListener('submit', async e => {
  e.preventDefault();
  const claim = $('#beliefClaim').value.trim();
  if (!claim) return;
  try {
    await saveBelief({
      claim,
      status: $('#beliefStatus').value,
      confidence: confidenceValue('#beliefConfidence', 0.5),
      source: $('#beliefSource').value.trim() || null
    });
    $('#beliefClaim').value = ''; $('#beliefSource').value = '';
  } catch (err) {
    $('#beliefList').prepend(el('p', 'empty', err.message));
  }
});
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  if (safetyOpen) closeSafety();
  if (actionsOpen) closeActions();
  if (observabilityOpen) closeObservability();
  if (systemsOpen) closeSystems();
  if (stateOpen) closeState();
  if (managerOpen) closeManager();
});

const savedCtx = Number(localStorage.getItem('nm_ctx'));
$('#ctx').value = savedCtx && savedCtx !== 4096 ? Math.min(savedCtx, 262144) : 262144;
$('#initiativeMode').value = localStorage.getItem('nm_initiative_mode') || 'helpful';
restoreChatState();
loadModels(); loadStats(); loadRetrievalDefaults(); loadReminders(); loadInitiatives(); loadAwareness(); loadPendingActions().catch(() => {});
// The app is often left open overnight. Refresh the cheap, no-LLM reminder
// endpoint so yesterday's event disappears without requiring a chat turn or a
// full page reload.
setInterval(loadReminders, 60_000);
