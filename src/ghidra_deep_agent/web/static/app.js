"use strict";

// ---- DOM helpers -----------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// ---- State -----------------------------------------------------------------
const state = {
  config: { model: "", max_context_tokens: 200000, mcp_ok: false, db_ok: false },
  sessions: [],
  activeId: null,
  ws: null,
  running: false,
  runStart: 0,
  timer: null,
  inputTokens: 0,
  outputTokens: 0,
  activeTools: 0,
  nodes: {}, // run_id -> { el, start }
  nsContainers: {}, // checkpoint_ns -> container element
  currentEntry: null, // { el, body, raw }
  lastResponse: "",
  transcript: [],
  history: [],
  histIdx: 0,
};

// ---- Formatting ------------------------------------------------------------
function fmtTokens(n) {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return (n / 1000).toFixed(1) + "k";
  return (n / 1_000_000).toFixed(2) + "M";
}
function fmtDuration(secs) {
  if (secs < 1) return Math.round(secs * 1000) + "ms";
  if (secs < 60) return secs.toFixed(1) + "s";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}m${String(s).padStart(2, "0")}s`;
}
function fmtClock(secs) {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---- Config & status bar ---------------------------------------------------
async function loadConfig() {
  state.config = await (await fetch("/api/config")).json();
  $("#st-mcp").classList.toggle("ok", state.config.mcp_ok);
  $("#st-mcp").classList.toggle("bad", !state.config.mcp_ok);
  $("#st-db").classList.toggle("ok", state.config.db_ok);
  $("#st-db").classList.toggle("bad", !state.config.db_ok);
}

function renderStatus() {
  $("#st-tokens").textContent = `↑${fmtTokens(state.inputTokens)} ↓${fmtTokens(
    state.outputTokens
  )}`;
  $("#st-tools").textContent = `tools ${state.activeTools}`;
}

function renderContext(current) {
  const max = state.config.max_context_tokens || 1;
  const pct = Math.round((current / max) * 100);
  const e = $("#st-context");
  e.textContent = `ctx ${fmtTokens(current)} (${pct}%)`;
  e.classList.toggle("warn", pct >= 75 && pct < 85);
  e.classList.toggle("crit", pct >= 85);
}

function flash(text) {
  $("#st-flash").textContent = text;
  clearTimeout(flash._t);
  flash._t = setTimeout(() => ($("#st-flash").textContent = ""), 4000);
}

function setBusy(busy) {
  state.running = busy;
  $("#query").classList.toggle("busy", busy);
  $("#statusbar").classList.toggle("busy", busy);
  if (busy) {
    state.runStart = Date.now();
    $("#st-timer").textContent = "0:00";
    state.timer = setInterval(() => {
      $("#st-timer").textContent = fmtClock(
        Math.floor((Date.now() - state.runStart) / 1000)
      );
    }, 1000);
  } else if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

// ---- Sessions sidebar ------------------------------------------------------
async function loadSessions() {
  const data = await (await fetch("/api/sessions")).json();
  state.sessions = data.sessions || [];
  renderSessions();
}

function renderSessions() {
  const list = $("#session-list");
  list.innerHTML = "";
  for (const s of state.sessions) {
    const li = el("li");
    li.classList.toggle("active", s.session_id === state.activeId);
    const title = el("span", "s-title", s.title || s.binary_name);
    title.title = `${s.binary_name} · ${s.session_id}`;
    const del = el("span", "s-del", "✕");
    del.title = "Delete session";
    del.onclick = (e) => {
      e.stopPropagation();
      deleteSession(s.session_id);
    };
    li.append(title, del);
    li.onclick = () => openSession(s);
    list.append(li);
  }
}

async function createSession(binaryName) {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ binary_name: binaryName }),
  });
  const session = await res.json();
  state.sessions.unshift(session);
  renderSessions();
  openSession(session);
}

async function deleteSession(id) {
  await fetch(`/api/sessions/${id}`, { method: "DELETE" });
  state.sessions = state.sessions.filter((s) => s.session_id !== id);
  if (state.activeId === id) {
    state.activeId = null;
    closeWs();
    resetConversation();
    $("#active-binary").textContent = "No session";
    $("#active-meta").textContent = "";
    $("#query").disabled = true;
  }
  renderSessions();
}

async function openSession(session) {
  if (state.activeId === session.session_id) return;
  state.activeId = session.session_id;
  renderSessions();
  resetConversation();
  state.inputTokens = 0;
  state.outputTokens = 0;
  state.activeTools = 0;
  renderStatus();
  $("#active-binary").textContent = session.binary_name;
  $("#active-meta").textContent = `${state.config.model} · ${session.session_id.slice(0, 8)}`;
  $("#query").disabled = false;
  $("#query").focus();

  await loadHistory(session.session_id);
  connectWs(session.session_id);
}

async function loadHistory(id) {
  try {
    const data = await (await fetch(`/api/sessions/${id}/history`)).json();
    for (const m of data.messages || []) {
      if (m.role === "user") logUser(m.content);
      else {
        const entry = startAssistantEntry();
        entry.raw = m.content;
        renderAssistant(entry);
      }
    }
  } catch (e) {
    /* no history yet */
  }
}

// ---- WebSocket -------------------------------------------------------------
function closeWs() {
  if (state.ws) {
    state.ws.onclose = null;
    state.ws.close();
    state.ws = null;
  }
}

function connectWs(id) {
  closeWs();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/sessions/${id}/stream`);
  state.ws = ws;
  ws.onmessage = (ev) => handlePayload(JSON.parse(ev.data));
  ws.onclose = () => {
    if (state.ws === ws) state.ws = null;
  };
}

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
    return true;
  }
  flash("Not connected to a session.");
  return false;
}

// ---- Conversation rendering ------------------------------------------------
function resetConversation() {
  $("#conversation").innerHTML = "";
  $("#activity-tree").innerHTML = "";
  $("#thinking-body").textContent = "";
  state.nodes = {};
  state.nsContainers = {};
  state.currentEntry = null;
  state.lastResponse = "";
  state.transcript = [];
}

function logUser(text) {
  const entry = el("div", "entry");
  entry.append(el("div", "user", text));
  $("#conversation").append(entry);
  state.transcript.push("❯ " + text);
  scrollConversation();
}

function startAssistantEntry() {
  const entry = el("div", "entry");
  const body = el("div", "assistant-body");
  entry.append(body);
  $("#conversation").append(entry);
  const obj = { el: entry, body, raw: "" };
  state.currentEntry = obj;
  return obj;
}

function renderAssistant(entry) {
  const md = window.marked ? window.marked.parse(entry.raw) : entry.raw;
  entry.body.innerHTML = md;
  scrollConversation();
}

function scrollConversation() {
  const c = $("#conversation");
  c.scrollTop = c.scrollHeight;
}

// ---- Activity tree ---------------------------------------------------------
function containerForNs(ns) {
  if (!ns) return $("#activity-tree");
  if (state.nsContainers[ns]) return state.nsContainers[ns];
  const segs = ns.split("|");
  const parentNs = segs.slice(0, -1).join("|");
  const parent = containerForNs(parentNs);
  const box = el("div", "children");
  parent.append(box);
  state.nsContainers[ns] = box;
  return box;
}

function onToolStart(p) {
  const node = el("div", "node running");
  const label = el("div", "node-label");
  const icon = el("span", "icon", p.is_subagent ? "▸" : "•");
  label.append(icon);
  label.append(
    document.createTextNode(` ${p.name}${p.preview ? "  " + p.preview : ""}`)
  );
  node.append(label);
  containerForNs(p.checkpoint_ns).append(node);
  state.nodes[p.run_id] = { el: node, label, start: performance.now() };
  state.activeTools += 1;
  renderStatus();
}

function onToolEnd(p) {
  const n = state.nodes[p.run_id];
  if (!n) return;
  n.el.classList.remove("running");
  n.el.classList.add(p.error ? "err" : "ok");
  const dur = (performance.now() - n.start) / 1000;
  n.label.append(el("span", "dur", fmtDuration(dur)));
  if (p.error && p.snippet) {
    n.el.append(el("div", "err-snip", p.snippet));
  }
}

function onLlmThinking(p) {
  const box = containerForNs(p.checkpoint_ns);
  let ind = box.querySelector(":scope > .thinking-indicator");
  if (!ind) {
    ind = el("div", "node thinking-indicator");
    ind.textContent = "✷ thinking…";
    box.append(ind);
  }
  state.nodes["think:" + p.run_id] = { el: ind };
  showThinking(true);
}

function onLlmDone(p) {
  const n = state.nodes["think:" + p.run_id];
  if (n && n.el) n.el.remove();
}

// ---- Thinking panel --------------------------------------------------------
function showThinking(on) {
  $("#thinking").classList.toggle("hidden", !on);
}

function appendThinking(text) {
  const body = $("#thinking-body");
  let t = body.textContent + text;
  if (t.length > 3000) t = t.slice(t.length - 3000);
  body.textContent = t;
  body.scrollTop = body.scrollHeight;
}

// ---- Payload dispatch ------------------------------------------------------
function handlePayload(p) {
  switch (p.type) {
    case "tool_start":
      onToolStart(p);
      break;
    case "tool_end":
      onToolEnd(p);
      break;
    case "llm_thinking":
      onLlmThinking(p);
      break;
    case "llm_done":
      onLlmDone(p);
      break;
    case "token":
      if (!state.currentEntry) startAssistantEntry();
      state.currentEntry.raw += p.text;
      state.currentEntry.body.textContent = state.currentEntry.raw;
      appendThinking(p.text);
      scrollConversation();
      break;
    case "token_update":
      state.inputTokens += p.input;
      state.outputTokens += p.output;
      renderStatus();
      break;
    case "context_update":
      renderContext(p.current_input);
      break;
    case "tool_count":
      state.activeTools = Math.max(0, state.activeTools + p.delta);
      renderStatus();
      break;
    case "status_flash":
      flash(p.text);
      break;
    case "toast":
      showToast(p);
      break;
    case "agent_done":
      finishRun();
      break;
    case "cancelled":
      if (state.currentEntry) {
        renderAssistant(state.currentEntry);
        state.currentEntry.el.classList.add("cancelled");
      }
      flash("Run cancelled.");
      finishRun();
      break;
    case "error":
      if (!state.currentEntry) startAssistantEntry();
      state.currentEntry.raw = "✗ Error: " + p.message;
      state.currentEntry.el.classList.add("error");
      renderAssistant(state.currentEntry);
      finishRun();
      break;
  }
}

function finishRun() {
  if (state.currentEntry) {
    renderAssistant(state.currentEntry);
    if (state.currentEntry.raw) {
      state.lastResponse = state.currentEntry.raw;
      state.transcript.push(state.currentEntry.raw);
    }
    state.currentEntry = null;
  }
  showThinking(false);
  setBusy(false);
  $("#query").focus();
}

// ---- Toasts ----------------------------------------------------------------
function showToast(p) {
  const t = el("div", "toast " + (p.severity || "information"));
  if (p.title) t.append(el("div", "t-title", p.title));
  t.append(el("div", "t-msg", p.message));
  $("#toasts").append(t);
  setTimeout(() => t.remove(), 6000);
}

// ---- Slash commands & submit -----------------------------------------------
function handleSubmit(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  state.history.push(trimmed);
  state.histIdx = state.history.length;

  if (trimmed.startsWith("/")) {
    const cmd = trimmed.split(/\s+/)[0].toLowerCase();
    if (cmd === "/clear") {
      resetConversation();
      flash("Cleared.");
      return;
    }
    if (cmd === "/yank") {
      doYank();
      return;
    }
    if (cmd === "/help") {
      $("#help-modal").classList.remove("hidden");
      return;
    }
    if (cmd === "/quit") {
      closeWs();
      flash("Disconnected.");
      return;
    }
    if (cmd !== "/compact") {
      flash("Unknown command: " + cmd);
      return;
    }
    // /compact falls through to the server.
  }

  if (state.running) {
    flash("Agent still running — please wait.");
    return;
  }
  logUser(trimmed);
  $("#activity-tree").innerHTML = "";
  state.nodes = {};
  state.nsContainers = {};
  if (send({ type: "query", text: trimmed })) {
    setBusy(true);
  }
}

function doYank() {
  if (!state.lastResponse) {
    flash("Nothing to copy yet.");
    return;
  }
  navigator.clipboard.writeText(state.lastResponse).then(
    () => flash("Response copied."),
    () => flash("Copy failed.")
  );
}

// ---- Binary picker ---------------------------------------------------------
async function openBinaryPicker() {
  const modal = $("#binary-modal");
  const list = $("#binary-list");
  const err = $("#binary-error");
  list.innerHTML = "";
  err.classList.add("hidden");
  modal.classList.remove("hidden");
  try {
    const res = await fetch("/api/programs");
    const data = await res.json();
    if (data.error) {
      err.textContent = data.error;
      err.classList.remove("hidden");
      return;
    }
    for (const name of data.programs || []) {
      const li = el("li", null, name);
      li.onclick = () => {
        modal.classList.add("hidden");
        createSession(name);
      };
      list.append(li);
    }
    if (!(data.programs || []).length) {
      err.textContent = "No open programs in Ghidra.";
      err.classList.remove("hidden");
    }
  } catch (e) {
    err.textContent = "Failed to reach Ghidra: " + e;
    err.classList.remove("hidden");
  }
}

// ---- Binary upload ---------------------------------------------------------
async function uploadBinary() {
  const input = $("#upload-file");
  const status = $("#upload-status");
  const file = input.files && input.files[0];
  if (!file) {
    status.textContent = "Choose a file first.";
    return;
  }
  const btn = $("#upload-btn");
  btn.disabled = true;
  status.textContent = `Importing ${file.name}… (this can take a moment)`;
  try {
    // Send the file as the raw request body with the name in the query string
    // (the server reads request.body() — no multipart parsing needed).
    const res = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok) {
      status.textContent = `Imported "${data.name}" into ${data.repository}. It's in the shared repo — check it out in Ghidra to analyze.`;
      input.value = "";
    } else {
      status.textContent = `Import failed (${res.status}): ${data.error || "unknown error"}`;
    }
  } catch (e) {
    status.textContent = "Upload failed: " + e;
  } finally {
    btn.disabled = false;
  }
}

// ---- Wiring ----------------------------------------------------------------
function init() {
  loadConfig();
  loadSessions();

  $("#input-row").addEventListener("submit", (e) => {
    e.preventDefault();
    const inp = $("#query");
    const val = inp.value;
    inp.value = "";
    handleSubmit(val);
  });

  $("#query").addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp") {
      if (state.histIdx > 0) {
        state.histIdx -= 1;
        e.target.value = state.history[state.histIdx] || "";
      }
      e.preventDefault();
    } else if (e.key === "ArrowDown") {
      if (state.histIdx < state.history.length) {
        state.histIdx += 1;
        e.target.value = state.history[state.histIdx] || "";
      }
      e.preventDefault();
    }
  });

  $("#new-session").onclick = openBinaryPicker;
  $("#upload-btn").onclick = uploadBinary;
  $("#binary-cancel").onclick = () => $("#binary-modal").classList.add("hidden");
  $("#toggle-tree").onclick = () => $("#panes").classList.toggle("hide-tree");
  $("#yank").onclick = doYank;
  $("#clear").onclick = () => {
    resetConversation();
    flash("Cleared.");
  };
  $("#help").onclick = () => $("#help-modal").classList.remove("hidden");
  $("#help-close").onclick = () => $("#help-modal").classList.add("hidden");
  $("#thinking-toggle").onclick = () => showThinking(false);

  document.addEventListener("keydown", (e) => {
    if (e.target.id === "query" && !["Escape"].includes(e.key)) return;
    if (e.key === "Escape") {
      if (!$("#help-modal").classList.contains("hidden")) {
        $("#help-modal").classList.add("hidden");
      } else if (!$("#binary-modal").classList.contains("hidden")) {
        $("#binary-modal").classList.add("hidden");
      } else if (state.running) {
        send({ type: "cancel" });
      }
    } else if (e.key === "F1") {
      e.preventDefault();
      $("#help-modal").classList.remove("hidden");
    } else if (e.ctrlKey && e.key === "l") {
      e.preventDefault();
      resetConversation();
    } else if (e.ctrlKey && e.key === "t") {
      e.preventDefault();
      $("#panes").classList.toggle("hide-tree");
    } else if (e.ctrlKey && e.key === "y") {
      e.preventDefault();
      doYank();
    }
  });
}

window.addEventListener("DOMContentLoaded", init);
