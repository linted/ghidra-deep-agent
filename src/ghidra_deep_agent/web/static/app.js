"use strict";

// ---- DOM helpers -----------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// Parse markdown, then syntax-highlight fenced code blocks in-place. marked v18
// dropped the `highlight` option, so we post-process `pre code` with highlight.js
// (both vendored). `target` is the element whose innerHTML was just set.
function renderInto(target, src) {
  target.innerHTML = window.marked ? window.marked.parse(src) : src;
  if (!window.hljs) return;
  for (const block of target.querySelectorAll("pre code")) {
    try {
      window.hljs.highlightElement(block);
    } catch {
      /* leave the block unhighlighted */
    }
  }
}

// ---- State -----------------------------------------------------------------
const state = {
  config: { model: "", max_context_tokens: 200000, mcp_ok: false, db_ok: false },
  sessions: [],
  sessionFilter: "",
  activeId: null,
  ws: null,
  wsState: "offline", // offline | connecting | connected | reconnecting
  reconnect: { attempts: 0, timer: null },
  errorsOnly: false,
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
function fmtRelative(iso) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const secs = Math.max(0, (Date.now() - t) / 1000);
  if (secs < 60) return "just now";
  if (secs < 3600) return Math.floor(secs / 60) + "m ago";
  if (secs < 86400) return Math.floor(secs / 3600) + "h ago";
  return Math.floor(secs / 86400) + "d ago";
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
  const q = state.sessionFilter.trim().toLowerCase();
  let shown = 0;
  for (const s of state.sessions) {
    const label = (s.title || s.binary_name || "").toLowerCase();
    if (q && !label.includes(q) && !s.binary_name.toLowerCase().includes(q)) {
      continue;
    }
    shown += 1;
    const li = el("li");
    li.classList.toggle("active", s.session_id === state.activeId);

    const main = el("div", "s-main");
    const title = el("span", "s-title", s.title || s.binary_name);
    title.title = "Double-click to rename";
    title.ondblclick = (e) => {
      e.stopPropagation();
      beginRename(li, s);
    };
    const meta = el("div", "s-meta");
    const when = fmtRelative(s.updated_at);
    meta.textContent = when ? `${s.binary_name} · ${when}` : s.binary_name;
    main.append(title, meta);

    const actions = el("div", "s-actions");
    const ren = el("span", "s-ren", "✎");
    ren.title = "Rename session";
    ren.onclick = (e) => {
      e.stopPropagation();
      beginRename(li, s);
    };
    const del = el("span", "s-del", "✕");
    del.title = "Delete session";
    del.onclick = (e) => {
      e.stopPropagation();
      deleteSession(s.session_id);
    };
    actions.append(ren, del);

    li.append(main, actions);
    li.onclick = () => openSession(s);
    list.append(li);
  }
  if (!shown) {
    const empty = el("li", "s-empty muted", q ? "No matches" : "No sessions yet");
    list.append(empty);
  }
}

// Swap a session's title for an inline editor; PATCH on commit.
function beginRename(li, session) {
  const main = li.querySelector(".s-main");
  const titleEl = main.querySelector(".s-title");
  const input = el("input", "s-rename");
  input.value = session.title || session.binary_name;
  titleEl.replaceWith(input);
  input.focus();
  input.select();

  let done = false;
  const commit = async (save) => {
    if (done) return;
    done = true;
    const next = input.value.trim();
    if (save && next && next !== session.title) {
      await renameSession(session.session_id, next);
    } else {
      renderSessions();
    }
  };
  input.onclick = (e) => e.stopPropagation();
  input.onkeydown = (e) => {
    if (e.key === "Enter") commit(true);
    else if (e.key === "Escape") commit(false);
  };
  input.onblur = () => commit(true);
}

async function renameSession(id, title) {
  try {
    const res = await fetch(`/api/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    const data = await res.json();
    if (res.ok) {
      const s = state.sessions.find((x) => x.session_id === id);
      if (s) s.title = data.title;
    } else {
      flash(data.error || "Rename failed.");
    }
  } catch (e) {
    flash("Rename failed.");
  }
  renderSessions();
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
  state.reconnect.attempts = 0;
  renderSessions();
  resetConversation();
  state.inputTokens = 0;
  state.outputTokens = 0;
  state.activeTools = 0;
  renderStatus();
  $("#active-binary").textContent = session.binary_name;
  $("#active-meta").textContent = `${state.config.model} · ${session.session_id.slice(0, 8)}`;

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
const RECONNECT_MAX_MS = 10000;

// Reflect the live-connection state in the status bar and gate the input on it.
function setConn(s) {
  state.wsState = s;
  const dot = $("#st-conn");
  dot.classList.toggle("ok", s === "connected");
  dot.classList.toggle("bad", s === "offline" || s === "reconnecting");
  dot.title =
    s === "connected"
      ? "Live connection"
      : s === "reconnecting"
        ? "Reconnecting…"
        : s === "connecting"
          ? "Connecting…"
          : "Disconnected";
  // Only re-enable input when a session is active and the socket is up.
  if (state.activeId) $("#query").disabled = s !== "connected";
}

function closeWs() {
  clearTimeout(state.reconnect.timer);
  state.reconnect.timer = null;
  if (state.ws) {
    state.ws.onclose = null;
    state.ws.close();
    state.ws = null;
  }
  setConn("offline");
}

function connectWs(id) {
  // Intentional teardown of any prior socket without flipping us to "offline".
  clearTimeout(state.reconnect.timer);
  if (state.ws) {
    state.ws.onclose = null;
    state.ws.close();
    state.ws = null;
  }
  setConn(state.reconnect.attempts ? "reconnecting" : "connecting");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/sessions/${id}/stream`);
  state.ws = ws;
  ws.onopen = () => {
    if (state.reconnect.attempts) flash("Reconnected.");
    state.reconnect.attempts = 0;
    setConn("connected");
    $("#query").focus();
  };
  ws.onmessage = (ev) => handlePayload(JSON.parse(ev.data));
  ws.onclose = () => {
    if (state.ws !== ws) return; // superseded by a newer socket
    state.ws = null;
    // The server cancels any in-flight run when the socket drops.
    if (state.running) {
      setBusy(false);
      showThinking(false);
      flash("Connection lost — run interrupted.");
    }
    scheduleReconnect(id);
  };
}

function scheduleReconnect(id) {
  if (state.activeId !== id) return; // user moved on; don't reconnect
  setConn("reconnecting");
  const delay = Math.min(RECONNECT_MAX_MS, 1000 * 2 ** state.reconnect.attempts);
  state.reconnect.attempts += 1;
  state.reconnect.timer = setTimeout(() => {
    if (state.activeId === id) connectWs(id);
  }, delay);
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
  renderInto(entry.body, entry.raw);
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
    const snip = el("div", "err-snip", p.snippet);
    snip.title = "Click to expand";
    snip.onclick = (e) => {
      e.stopPropagation();
      snip.classList.toggle("expanded");
    };
    n.el.append(snip);
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
  loadLanguages();
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
// {languageID: [compilerSpecID, ...]} for the processor picker; loaded lazily.
let langSpecs = {};

async function loadLanguages() {
  const datalist = $("#ghidra-langs");
  if (!datalist || datalist.dataset.loaded) return;
  try {
    const data = await (await fetch("/api/languages")).json();
    for (const lang of data.languages || []) {
      langSpecs[lang.id] = lang.compilerSpecs || [];
      const opt = el("option");
      opt.value = lang.id; // datalist matches on value, not text content
      datalist.append(opt);
    }
    datalist.dataset.loaded = "1";
  } catch (e) {
    // Non-fatal: auto-detect still works; raw-binary import just lacks the list.
  }
}

// Populate the compiler-spec dropdown for the chosen processor.
function onProcessorChange() {
  const proc = $("#upload-processor").value.trim();
  const sel = $("#upload-cspec");
  const specs = langSpecs[proc] || [];
  sel.innerHTML = "";
  sel.append(el("option", null, "default"));
  for (const id of specs) {
    if (id !== "default") sel.append(el("option", null, id));
  }
}

function uploadBinary() {
  const input = $("#upload-file");
  const status = $("#upload-status");
  const bar = $("#upload-progress");
  const file = input.files && input.files[0];
  if (!file) {
    status.textContent = "Choose a file first.";
    return;
  }
  const btn = $("#upload-btn");
  btn.disabled = true;

  // Send the file as the raw request body with the name in the query string (the
  // server reads request.body() — no multipart parsing needed). Optional
  // loader/processor/cspec/base hints let raw/headerless binaries import. XHR
  // (not fetch) gives us upload progress events for the bar.
  const params = new URLSearchParams({ name: file.name });
  const hints = {
    processor: $("#upload-processor").value.trim(),
    cspec: $("#upload-cspec").value.trim(),
    base: $("#upload-base").value.trim(),
    loader: $("#upload-loader").value.trim(),
  };
  for (const [k, v] of Object.entries(hints)) if (v) params.set(k, v);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/upload?${params.toString()}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");

  bar.classList.remove("hidden");
  bar.removeAttribute("value"); // indeterminate until the first progress event
  status.textContent = `Uploading ${file.name}…`;

  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    bar.value = pct;
    status.textContent =
      pct < 100
        ? `Uploading ${file.name}… ${pct}%`
        : `Importing ${file.name}… (this can take a moment)`;
  };
  xhr.upload.onload = () => {
    bar.removeAttribute("value"); // back to indeterminate while the server imports
    status.textContent = `Importing ${file.name}… (this can take a moment)`;
  };
  xhr.onload = () => {
    bar.classList.add("hidden");
    btn.disabled = false;
    let data = {};
    try {
      data = JSON.parse(xhr.responseText);
    } catch {
      /* keep empty */
    }
    if (xhr.status >= 200 && xhr.status < 300) {
      status.textContent = `Imported "${data.name}" into ${data.repository}. It's in the shared repo — check it out in Ghidra to analyze.`;
      input.value = "";
    } else {
      status.textContent = `Import failed (${xhr.status}): ${data.error || "unknown error"}`;
    }
  };
  xhr.onerror = () => {
    bar.classList.add("hidden");
    btn.disabled = false;
    status.textContent = "Upload failed: network error.";
  };
  xhr.send(file);
}

// ---- Resizable activity pane -----------------------------------------------
const ACTIVITY_MIN = 200;
const ACTIVITY_MAX = 640;

function setActivityWidth(px) {
  const w = Math.round(Math.min(ACTIVITY_MAX, Math.max(ACTIVITY_MIN, px)));
  $("#panes").style.setProperty("--activity-w", w + "px");
  return w;
}

function initPaneResize() {
  const saved = parseInt(localStorage.getItem("activityWidth") || "", 10);
  if (!Number.isNaN(saved)) setActivityWidth(saved);

  const handle = $("#pane-resize");
  const panes = $("#panes");
  let dragging = false;

  const onMove = (e) => {
    if (!dragging) return;
    // Width measured from the right edge of the panes to the pointer.
    const w = setActivityWidth(panes.getBoundingClientRect().right - e.clientX);
    localStorage.setItem("activityWidth", String(w));
  };
  const stop = () => {
    dragging = false;
    document.body.classList.remove("resizing");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", stop);
  };
  handle.addEventListener("pointerdown", (e) => {
    if (panes.classList.contains("hide-tree")) return;
    dragging = true;
    e.preventDefault();
    document.body.classList.add("resizing");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", stop);
  });
}

// ---- Wiring ----------------------------------------------------------------
function init() {
  setConn("offline");
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
  $("#upload-processor").addEventListener("change", onProcessorChange);
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

  $("#session-filter").addEventListener("input", (e) => {
    state.sessionFilter = e.target.value;
    renderSessions();
  });

  $("#activity-filter").onclick = () => {
    state.errorsOnly = !state.errorsOnly;
    $("#activity-tree").classList.toggle("errors-only", state.errorsOnly);
    $("#activity-filter").textContent = state.errorsOnly ? "errors" : "all";
  };

  // Collapse/expand a subagent's children by clicking its label (event-delegated
  // so it covers nodes added later). A node is collapsible iff a `.children` box
  // immediately follows it; CSS draws the caret and hides the children.
  $("#activity-tree").addEventListener("click", (e) => {
    const label = e.target.closest(".node-label");
    if (!label) return;
    const node = label.parentElement;
    const next = node.nextElementSibling;
    if (next && next.classList.contains("children")) {
      node.classList.toggle("collapsed");
    }
  });

  initPaneResize();

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
