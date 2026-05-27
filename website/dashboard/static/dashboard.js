/**
 * Hermes Web Dashboard — single-page app
 *
 * Connects to the Hermes gateway via WebSocket (JSON-RPC 2.0),
 * sends messages via gateway.submit, and renders streaming responses.
 * File operations go via REST: POST /upload, GET /browse, GET /download.
 */

"use strict";

// ── State ────────────────────────────────────────────────────────────
let ws = null;
let wsReady = false;
let msgCount = 0;

// uploadedFiles: [{ file_id, filename, size, path }]
const uploadedFiles = [];

// attachedFiles: files attached to the current draft message [{ file_id, filename, path }]
const attachedFiles = [];

// selectedBrowsePath: current path in file browser modal
let selectedBrowsePath = "";

// ── DOM refs ─────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const messagesEl     = $("messages");
const inputEl        = $("input");
const sendBtn        = $("btn-send");
const attachBtn      = $("btn-attach");
const uploadChips    = $("upload-chips");
const attachedFilesEl= $("attached-files");
const statusBadge    = $("status-badge");
const typingEl       = $("typing-indicator");
const modalFiles     = $("modal-files");
const modalUploads   = $("modal-uploads");
const fileListEl     = $("file-list");
const currentPathEl  = $("current-path");
const uploadListEl   = $("upload-list");
const dropzoneEl     = $("dropzone");
const fileInputEl    = $("file-input");

// ── WebSocket ───────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws`;

  ws = new WebSocket(url);
  ws.onopen    = onWSOpen;
  ws.onclose   = onWSClose;
  ws.onerror   = onWSError;
  ws.onmessage = onWSMessage;
}

function onWSOpen() {
  setStatus("connected");
  wsReady = true;
  // gateway.ready is sent by server immediately after connect
}

function onWSClose() {
  wsReady = false;
  setStatus("disconnected");
  // Reconnect after 3s
  setTimeout(connectWS, 3000);
}

function onWSError() {
  setStatus("error");
}

function sendWS(obj) {
  if (ws && wsReady && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ── JSON-RPC helpers ─────────────────────────────────────────────────
let pendingRPCs = {};

function nextId() { return ++msgCount; }

function callGateway(method, params) {
  return new Promise((resolve, reject) => {
    const id = nextId();
    pendingRPCs[id] = { resolve, reject };
    sendWS({ jsonrpc: "2.0", id, method, params });
    // Timeout after 60s
    setTimeout(() => {
      if (pendingRPCs[id]) {
        delete pendingRPCs[id];
        reject(new Error("Request timeout"));
      }
    }, 60000);
  });
}

function notifyGateway(method, params) {
  sendWS({ jsonrpc: "2.0", method, params });
}

// ── WS message dispatcher ─────────────────────────────────────────────
function onWSMessage(event) {
  let msg;
  try { msg = JSON.parse(event.data); } catch { return; }

  // RPC response
  if (msg.jsonrpc === "2.0" && msg.id != null) {
    const pending = pendingRPCs[msg.id];
    if (pending) {
      delete pendingRPCs[msg.id];
      if (msg.error) pending.reject(new Error(msg.error.message || JSON.stringify(msg.error)));
      else pending.resolve(msg.result);
    }
    return;
  }

  // Gateway event
  if (msg.method === "event" && msg.params) {
    handleGatewayEvent(msg.params);
    return;
  }
}

// ── Gateway events ───────────────────────────────────────────────────
function handleGatewayEvent(params) {
  switch (params.type) {
    case "gateway.ready":
      setStatus("connected");
      break;

    case "agent.started":
      showTyping(true);
      break;

    case "agent.done":
      showTyping(false);
      break;

    case "run.text_delta":
      appendDelta(params.payload || "");
      break;

    case "run.tool_use":
      appendToolUse(params.payload || {});
      break;

    case "run.error":
      showTyping(false);
      appendMsg("system", `Error: ${params.payload?.error || "Unknown error"}`);
      break;

    case "approval.required":
      // Show approval banner
      appendApprovalBanner(params.payload || {});
      break;

    case "session.reset":
      messagesEl.innerHTML = "";
      break;
  }
}

// ── Approval ─────────────────────────────────────────────────────────
function appendApprovalBanner(payload) {
  const { question, session_key } = payload;
  const id = `approval-${Date.now()}`;
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.id = id;
  el.innerHTML = `
    <div class="msg-avatar">⚠</div>
    <div class="msg-body">
      <div style="color:var(--accent);font-weight:600;margin-bottom:8px;">Approval Required</div>
      <div style="margin-bottom:12px;">${escapeHtml(question || "An action requires your approval.")}</div>
      <div style="display:flex;gap:8px;">
        <button class="action-btn" onclick="sendApproval('${session_key}', 'approve', '${id}')">Approve</button>
        <button class="action-btn secondary" onclick="sendApproval('${session_key}', 'deny', '${id}')">Deny</button>
      </div>
    </div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// Global so inline onclick can find it
window.sendApproval = async function(sessionKey, choice, elId) {
  try {
    await callGateway("gateway.respond_approval", { session_key: sessionKey, choice });
    document.getElementById(elId)?.remove();
  } catch(e) {
    console.error("approval error", e);
  }
};

// ── Send message ────────────────────────────────────────────────────
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text && attachedFiles.length === 0) return;
  if (!wsReady) { alert("Not connected to Hermes. Waiting for connection..."); return; }

  const textContent = text || "";
  inputEl.value = "";
  autoResize(inputEl);

  // Build content array — text + file references
  const content = [];
  if (textContent) content.push({ type: "text", text: textContent });
  for (const f of attachedFiles) {
    content.push({ type: "file", file_id: f.file_id, filename: f.filename, path: f.path });
  }

  // Show user message immediately
  const userEl = appendMsg("user", textContent, attachedFiles.splice(0));
  userEl?.scrollIntoView({ behavior: "smooth" });

  showTyping(true);

  try {
    const result = await callGateway("gateway.submit", {
      message: { role: "user", content },
      streaming: true,
    });
    showTyping(false);
    if (result && result.error) {
      appendMsg("system", `Error: ${result.error}`);
    }
    // result is null — events stream in via WebSocket
  } catch(e) {
    showTyping(false);
    appendMsg("system", `Failed to send: ${e.message}`);
  }
}

// ── Append message / delta ───────────────────────────────────────────
let currentAssistantEl = null;
let currentDeltaBuf = "";

function appendMsg(role, text, fileChips = []) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  const avatar = role === "user" ? "🧑" : role === "assistant" ? "⚕" : "ℹ";
  el.innerHTML = `<div class="msg-avatar">${avatar}</div><div class="msg-body"></div>`;
  const body = el.querySelector(".msg-body");

  if (fileChips && fileChips.length > 0) {
    const chips = fileChips.map(f =>
      `<span class="file-chip">📎 <span class="chip-name">${escapeHtml(f.filename)}</span></span>`
    ).join("");
    body.innerHTML += chips + "\n";
  }

  if (text) body.innerHTML += escapeHtml(text).replace(/\n/g, "<br>");
  messagesEl.appendChild(el);
  scrollToBottom();
  return el;
}

function appendDelta(delta) {
  if (!currentAssistantEl) {
    currentAssistantEl = document.createElement("div");
    currentAssistantEl.className = "msg assistant";
    currentAssistantEl.innerHTML = `<div class="msg-avatar">⚕</div><div class="msg-body"></div>`;
    messagesEl.appendChild(currentAssistantEl);
  }
  currentDeltaBuf += delta;
  const body = currentAssistantEl.querySelector(".msg-body");
  // Simple markdown-ish rendering
  body.innerHTML = renderMarkdown(currentDeltaBuf);
  scrollToBottom();
}

function appendToolUse(payload) {
  currentAssistantEl = null; // force new message
  currentDeltaBuf = "";
  const name = payload.name || "tool";
  const args = payload.arguments || {};
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="msg-avatar">⚙</div><div class="msg-body"><div class="tool-use"><span class="tool-name">${escapeHtml(name)}</span><div class="tool-args">${escapeHtml(JSON.stringify(args, null, 2))}</div></div></div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function flushAssistant() {
  if (currentAssistantEl) {
    currentAssistantEl = null;
    currentDeltaBuf = "";
  }
}

// ── Markdown-lite renderer ────────────────────────────────────────────
function renderMarkdown(text) {
  let out = escapeHtml(text);
  // Code blocks
  out = out.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`);
  // Inline code
  out = out.replace(/`([^`]+)`/g, `<code>$1</code>`);
  // Bold
  out = out.replace(/\*\*([^*]+)\*\*/g, `<strong>$1</strong>`);
  // Italic
  out = out.replace(/\*([^*]+)\*/g, `<em>$1</em>`);
  // Line breaks
  out = out.replace(/\n/g, "<br>");
  return out;
}

// ── Upload chips UI ──────────────────────────────────────────────────
function rebuildUploadChips() {
  if (attachedFiles.length === 0) {
    uploadChips.classList.add("hidden");
    attachedFilesEl.innerHTML = "";
    return;
  }
  uploadChips.classList.remove("hidden");
  attachedFilesEl.innerHTML = attachedFiles.map((f, i) => `
    <span class="file-chip" id="chip-${i}">
      📎 <span class="chip-name">${escapeHtml(f.filename)}</span>
      <button class="chip-rm" onclick="removeAttached(${i})">✕</button>
    </span>`).join("");
}

window.removeAttached = function(idx) {
  attachedFiles.splice(idx, 1);
  rebuildUploadChips();
};

// ── File Upload ──────────────────────────────────────────────────────
async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);

  let res;
  try {
    res = await fetch("/upload", { method: "POST", body: form });
  } catch(e) {
    alert("Upload failed: " + e.message);
    return;
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Upload failed" }));
    alert("Upload error: " + (err.error || res.statusText));
    return;
  }

  const info = await res.json();
  uploadedFiles.push(info);
  rebuildUploadList();
  return info;
}

async function uploadThenAttach(file) {
  const info = await uploadFile(file);
  if (info) {
    attachedFiles.push(info);
    rebuildUploadChips();
  }
}

// ── Upload list in modal ─────────────────────────────────────────────
function rebuildUploadList() {
  if (uploadedFiles.length === 0) {
    uploadListEl.innerHTML = '<p style="text-align:center;color:var(--text-muted);font-size:13px;padding:16px;">No uploads yet. Drop files above.</p>';
    return;
  }
  uploadListEl.innerHTML = uploadedFiles.map((f, i) => `
    <div class="upload-item">
      <span style="font-size:16px;">📄</span>
      <span class="ui-name">${escapeHtml(f.filename)}</span>
      <span class="ui-size">${formatBytes(f.size)}</span>
      <button class="ui-rm" onclick="removeUpload(${i})">✕</button>
      <button class="ui-rm" onclick="attachUpload(${i})" title="Attach to message">⬆</button>
      <button class="ui-rm" onclick="downloadFile('${f.path.replace(/'/g, "\\'")}')" title="Download">⬇</button>
    </div>`).join("");
}

window.removeUpload = function(idx) {
  uploadedFiles.splice(idx, 1);
  rebuildUploadList();
};

window.attachUpload = function(idx) {
  const f = uploadedFiles[idx];
  if (f && !attachedFiles.find(a => a.file_id === f.file_id)) {
    attachedFiles.push(f);
    rebuildUploadChips();
  }
  closeModalUploads();
};

window.downloadFile = function(path) {
  const url = `/download?path=${encodeURIComponent(path)}`;
  window.open(url, "_blank");
};

// ── File Browser ────────────────────────────────────────────────────
async function openFileBrowser() {
  selectedBrowsePath = "";
  modalFiles.classList.remove("hidden");
  await browse(selectedBrowsePath);
}

async function browse(path) {
  fileListEl.innerHTML = '<div class="loading">Loading...</div>';
  let res;
  try {
    res = await fetch(`/browse?path=${encodeURIComponent(path || "~")}`);
  } catch(e) {
    fileListEl.innerHTML = '<div class="loading">Failed to load. Is Hermes running?</div>';
    return;
  }
  const data = await res.json();
  if (data.error) {
    fileListEl.innerHTML = `<div class="loading" style="color:var(--danger);">Error: ${escapeHtml(data.error)}</div>`;
    return;
  }
  selectedBrowsePath = data.path;
  currentPathEl.textContent = data.path;
  renderFileList(data.entries);
}

function renderFileList(entries) {
  if (entries.length === 0) {
    fileListEl.innerHTML = '<div class="loading">Empty folder</div>';
    return;
  }
  // Dirs first, then files
  entries = [...entries].sort((a, b) => {
    if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  fileListEl.innerHTML = entries.map((e, i) => `
    <div class="file-entry" data-index="${i}" onclick="onFileEntryClick(event, ${i})">
      <span class="fe-icon">${e.is_dir ? "📁" : "📄"}</span>
      <div class="fe-info">
        <div class="fe-name">${escapeHtml(e.name)}</div>
        <div class="fe-meta">${e.is_dir ? "Folder" : formatBytes(e.size)}</div>
      </div>
      <div class="fe-actions">
        ${e.is_dir ? "" : `<button class="icon-btn" onclick="event.stopPropagation(); downloadFile('${e.path.replace(/'/g, "\\'")}')" title="Download">⬇</button>`}
        <button class="icon-btn" onclick="event.stopPropagation(); attachFileByPath('${e.path.replace(/'/g, "\\'")}')" title="Attach to message">⬆</button>
      </div>
    </div>`).join("");

  // Store entries for click handler
  fileListEl._entries = entries;
}

window.onFileEntryClick = async function(event, index) {
  const entries = fileListEl._entries || [];
  const entry = entries[index];
  if (!entry) return;
  if (entry.is_dir) {
    await browse(entry.path);
  } else {
    // Select it visually
    document.querySelectorAll(".file-entry").forEach(el => el.classList.remove("selected"));
    event.currentTarget.classList.add("selected");
    window._selectedFile = entry;
  }
};

window.attachFileByPath = async function(path) {
  // Upload the file then attach it
  // For large files, we just create a minimal entry referencing the path
  // (the agent will read it directly via read_file at that path)
  const filename = path.split("/").pop();
  const fakeId = "path-" + Date.now();
  const entry = { file_id: fakeId, filename, size: 0, path };
  attachedFiles.push(entry);
  rebuildUploadChips();
  closeModalFiles();
};

window.goUp = async function() {
  const parts = selectedBrowsePath.split("/");
  parts.pop();
  const parent = parts.join("/") || "/";
  await browse(parent);
};

window.selectBrowseDir = async function() {
  if (!selectedBrowsePath) return;
  // Notify the agent about this directory
  if (wsReady) {
    notifyGateway("gateway.submit", {
      message: {
        role: "user",
        content: [{ type: "text", text: `[System] User selected working directory: ${selectedBrowsePath}` }],
      },
      streaming: false,
    });
  }
  closeModalFiles();
};

function closeModalFiles() {
  modalFiles.classList.add("hidden");
  window._selectedFile = null;
}
function closeModalUploads() { modalUploads.classList.remove("hidden"); }

// ── Drag & drop ─────────────────────────────────────────────────────
function setupDragDrop() {
  const dz = dropzoneEl;
  dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag-over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
  dz.addEventListener("drop", async e => {
    e.preventDefault();
    dz.classList.remove("drag-over");
    for (const file of e.dataTransfer.files) {
      await uploadFile(file);
    }
    rebuildUploadList();
  });
  dz.addEventListener("click", () => fileInputEl.click());
  fileInputEl.addEventListener("change", async () => {
    for (const file of fileInputEl.files) await uploadFile(file);
    rebuildUploadList();
    fileInputEl.value = "";
  });
}

// ── Helpers ─────────────────────────────────────────────────────────
function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatBytes(n) {
  if (n === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(n) / Math.log(k));
  return parseFloat((n / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping(show) {
  if (show) {
    typingEl.classList.remove("hidden");
    scrollToBottom();
  } else {
    typingEl.classList.add("hidden");
    flushAssistant();
  }
}

function setStatus(state) {
  statusBadge.className = "badge " + state;
  statusBadge.textContent =
    state === "connected" ? "Connected" :
    state === "error" ? "Error" : "Disconnected";
}

function clearChat() {
  messagesEl.innerHTML = "";
  flushAssistant();
}

// ── Event wiring ─────────────────────────────────────────────────────
function init() {
  connectWS();

  // Send
  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  inputEl.addEventListener("input", () => autoResize(inputEl));

  // Attach
  attachBtn.addEventListener("click", () => {
    fileInputEl.click();
  });
  fileInputEl.addEventListener("change", async () => {
    for (const file of fileInputEl.files) await uploadThenAttach(file);
    fileInputEl.value = "";
  });

  // Header buttons
  $("btn-files").addEventListener("click", openFileBrowser);
  $("btn-uploads").addEventListener("click", () => {
    rebuildUploadList();
    modalUploads.classList.remove("hidden");
  });
  $("btn-clear").addEventListener("click", clearChat);

  // File modal
  $("btn-close-files").addEventListener("click", closeModalFiles);
  $("btn-path-up").addEventListener("click", window.goUp);
  $("btn-select-dir").addEventListener("click", window.selectBrowseDir);

  // Upload modal
  $("btn-close-uploads").addEventListener("click", () => modalUploads.classList.add("hidden"));
  $("btn-browse").addEventListener("click", () => fileInputEl.click());

  // Close modals on backdrop click
  modalFiles.addEventListener("click", e => { if (e.target === modalFiles) closeModalFiles(); });
  modalUploads.addEventListener("click", e => { if (e.target === modalUploads) modalUploads.classList.add("hidden"); });

  setupDragDrop();

  // Welcome message
  appendMsg("system", "Hermes dashboard loaded. Type a message or use 📁 to browse files.");
}

document.addEventListener("DOMContentLoaded", init);
