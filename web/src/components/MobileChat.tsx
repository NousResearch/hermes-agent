/**
 * MobileChat — a native, phone-first chat UI for the dashboard.
 *
 * Replaces the cramped terminal-in-the-browser on small screens with a real
 * messaging UI: streamed markdown bubbles, reasoning, tool cards, inline
 * approvals, image/PDF attachments, and a docked input bar.
 *
 * A turn is rendered as an ORDERED sequence of blocks (reasoning → text → tool
 * → text → …) so tool calls appear inline where they happened, never below the
 * final answer. Talks directly to the tui_gateway JSON-RPC WebSocket
 * (`/api/ws`): open → `gateway.ready` → `session.create`/`session.resume` →
 * `prompt.submit {session_id,text}`; events nested at params.payload.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Square, Plus, ChevronDown, Wrench, X, FileText, Mic, Loader2 } from "lucide-react";

import { Markdown } from "@/components/Markdown";
import { buildWsAuthParam, HERMES_BASE_PATH, api, fetchJSON } from "@/lib/api";

function greeting(): string {
  const h = new Date().getHours();
  if (h < 5) return "Good evening";
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}
const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);
function blobToDataUrl(b: Blob): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result || ""));
    r.onerror = rej;
    r.readAsDataURL(b);
  });
}

interface MsgItem { kind: "msg"; id: string; role: "user" | "assistant"; text: string; streaming: boolean; attachments?: { name: string; kind: "image" | "pdf"; thumb?: string }[] }
interface ReasoningItem { kind: "reasoning"; id: string; text: string; streaming: boolean }
interface ToolItem { kind: "tool"; id: string; toolId: string; name: string; context: string; args: string; result: string; done: boolean; durationS?: number }
interface StatusItem { kind: "status"; id: string; text: string }
interface ApprovalItem { kind: "approval"; id: string; approvalId: string; description: string; command: string; resolved?: "approve" | "deny" }
interface ErrorItem { kind: "error"; id: string; text: string }
type Item = MsgItem | ReasoningItem | ToolItem | StatusItem | ApprovalItem | ErrorItem;

interface Pending { id: string; name: string; kind: "image" | "pdf"; thumb?: string; status: "uploading" | "ready" | "error" }
type Conn = "connecting" | "ready" | "reconnecting" | "closed";

let _seq = 0;
const uid = () => `m${Date.now().toString(36)}-${(_seq++).toString(36)}`;
const stripDataUrl = (d: string) => d.replace(/^data:[^;]*;base64,/, "");

// Render any value as readable text (tool args/results arrive as objects).
function asText(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}
function gatewayWsUrl(k: string, val: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${HERMES_BASE_PATH}/api/ws?${new URLSearchParams({ [k]: val }).toString()}`;
}
function historyToItems(messages: unknown[]): Item[] {
  const out: Item[] = [];
  for (const m of messages) {
    if (!m || typeof m !== "object") continue;
    const rec = m as Record<string, unknown>;
    if (rec.role !== "user" && rec.role !== "assistant") continue;
    const text = typeof rec.text === "string" ? rec.text : asText(rec.content);
    if (!text) continue;
    out.push({ kind: "msg", id: uid(), role: rec.role, text, streaming: false });
  }
  return out;
}

export default function MobileChat({ profile, resume }: { profile?: string; resume?: string | null }) {
  const [items, setItems] = useState<Item[]>([]);
  const [draft, setDraft] = useState("");
  const [conn, setConn] = useState<Conn>("connecting");
  const [running, setRunning] = useState(false);
  const [model, setModel] = useState("");
  const [pending, setPending] = useState<Pending[]>([]);
  const [userName, setUserName] = useState("");
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const mrRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const sidRef = useRef("");
  const rpcId = useRef(1);
  const attachRpc = useRef<Record<number, string>>({});
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const taRef = useRef<HTMLTextAreaElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmounting = useRef(false);
  const nearBottom = useRef(true);
  // Per-turn cursors into the ordered block sequence.
  const curText = useRef<string | null>(null);
  const curReason = useRef<string | null>(null);
  const turnHadText = useRef(false);

  const push = useCallback((it: Item) => setItems((p) => [...p, it]), []);
  const patch = useCallback((id: string, fn: (it: Item) => Item) => setItems((p) => p.map((it) => (it.id === id ? fn(it) : it))), []);

  const sendRpc = useCallback((method: string, params: Record<string, unknown>): number => {
    const ws = wsRef.current;
    const id = rpcId.current++;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
    return id;
  }, []);

  const onEvent = useCallback(
    (type: string, p: Record<string, unknown>) => {
      const text = typeof p.text === "string" ? p.text : "";
      switch (type) {
        case "message.start":
          curText.current = null; curReason.current = null; turnHadText.current = false;
          break;
        case "reasoning.delta":
        case "thinking.delta": {
          if (!text) break;
          if (!curReason.current) { const id = uid(); curReason.current = id; push({ kind: "reasoning", id, text: "", streaming: true }); }
          patch(curReason.current, (it) => (it.kind === "reasoning" ? { ...it, text: it.text + text } : it));
          break;
        }
        case "message.delta": {
          // Answer text began → close the reasoning block (collapses it).
          if (curReason.current) { patch(curReason.current, (it) => (it.kind === "reasoning" ? { ...it, streaming: false } : it)); curReason.current = null; }
          if (!curText.current) { const id = uid(); curText.current = id; turnHadText.current = true; push({ kind: "msg", id, role: "assistant", text: "", streaming: true }); }
          patch(curText.current, (it) => (it.kind === "msg" ? { ...it, text: it.text + text } : it));
          break;
        }
        case "message.complete": {
          // Whole turn answered only at completion (no streamed deltas).
          if (!turnHadText.current && text) push({ kind: "msg", id: uid(), role: "assistant", text, streaming: false });
          if (curText.current) patch(curText.current, (it) => (it.kind === "msg" ? { ...it, streaming: false } : it));
          if (curReason.current) patch(curReason.current, (it) => (it.kind === "reasoning" ? { ...it, streaming: false } : it));
          curText.current = null; curReason.current = null;
          setRunning(false);
          break;
        }
        case "tool.start": {
          // Close the current text segment so the next text starts a NEW
          // block AFTER this tool — keeps blocks in chronological order.
          if (curText.current) { patch(curText.current, (it) => (it.kind === "msg" ? { ...it, streaming: false } : it)); curText.current = null; }
          push({ kind: "tool", id: uid(), toolId: String(p.tool_id ?? uid()), name: String(p.name ?? "tool"), context: String(p.context ?? ""), args: asText(p.args ?? p.args_text), result: "", done: false });
          break;
        }
        case "tool.complete": {
          const toolId = String(p.tool_id ?? "");
          const result = asText(p.result ?? p.summary);
          const args = asText(p.args);
          const dur = typeof p.duration_s === "number" ? p.duration_s : undefined;
          setItems((prev) => prev.map((it) => (it.kind === "tool" && it.toolId === toolId && !it.done ? { ...it, result, args: args || it.args, done: true, durationS: dur } : it)));
          break;
        }
        case "status.update":
          if (text) push({ kind: "status", id: uid(), text });
          break;
        case "approval.request":
          push({ kind: "approval", id: uid(), approvalId: String(p.approval_id ?? ""), description: String(p.description ?? p.action ?? "Approve this action?"), command: String(p.command ?? "") });
          break;
        case "error":
          push({ kind: "error", id: uid(), text: text || "Something went wrong." });
          setRunning(false);
          break;
        case "session.info":
          if (typeof p.model === "string") setModel(p.model);
          if (p.running === false) setRunning(false);
          break;
      }
    },
    [push, patch],
  );

  const connect = useCallback(async () => {
    let k = "token", v = "";
    try { [k, v] = await buildWsAuthParam(); } catch { setConn("reconnecting"); return; }
    if (unmounting.current) return;
    const ws = new WebSocket(gatewayWsUrl(k, v));
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      for (const line of ev.data.split("\n")) {
        const s = line.trim();
        if (!s) continue;
        let f: Record<string, unknown>;
        try { f = JSON.parse(s); } catch { continue; }
        if (typeof f.id === "number" && attachRpc.current[f.id]) {
          const aid = attachRpc.current[f.id]; delete attachRpc.current[f.id];
          const ok = !f.error;
          setPending((prev) => prev.map((a) => (a.id === aid ? { ...a, status: ok ? "ready" : "error" } : a)));
          continue;
        }
        if (f.result && typeof f.result === "object") {
          const res = f.result as Record<string, unknown>;
          if (typeof res.session_id === "string" && !sidRef.current) {
            sidRef.current = res.session_id; setConn("ready");
            const info = res.info as Record<string, unknown> | undefined;
            if (info && typeof info.model === "string") setModel(info.model);
            if (Array.isArray(res.messages) && res.messages.length > 0) setItems(historyToItems(res.messages as unknown[]));
          }
          continue;
        }
        if (f.method === "event" && f.params) {
          const params = f.params as Record<string, unknown>;
          const type = String(params.type ?? "");
          if (type === "gateway.ready") {
            if (resume) sendRpc("session.resume", { session_id: resume, source: "mobile-web", cols: 72, ...(profile ? { profile } : {}) });
            else sendRpc("session.create", { source: "mobile-web", cols: 72, ...(profile ? { profile } : {}) });
            continue;
          }
          onEvent(type, (params.payload ?? {}) as Record<string, unknown>);
        }
      }
    };
    ws.onclose = () => {
      wsRef.current = null;
      if (unmounting.current) return;
      sidRef.current = ""; curText.current = null; curReason.current = null;
      setRunning(false); setConn("closed");
      reconnectTimer.current = setTimeout(() => void connect(), 1500);
    };
    ws.onerror = () => ws.close();
  }, [onEvent, sendRpc, profile, resume]);

  useEffect(() => {
    unmounting.current = false;
    void connect();
    return () => { unmounting.current = true; if (reconnectTimer.current) clearTimeout(reconnectTimer.current); wsRef.current?.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { const el = scrollRef.current; if (el && nearBottom.current) el.scrollTop = el.scrollHeight; }, [items, pending]);
  const onScroll = () => { const el = scrollRef.current; if (el) nearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 140; };
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const h = () => { const el = scrollRef.current; if (el && nearBottom.current) el.scrollTop = el.scrollHeight; };
    vv.addEventListener("resize", h);
    return () => vv.removeEventListener("resize", h);
  }, []);

  // Greeting name from the dashboard's auth identity (the basic-auth username).
  useEffect(() => {
    let cancelled = false;
    api.getAuthMe().then((me) => { if (!cancelled && me?.user_id) setUserName(me.user_id); }).catch(() => {});
    return () => { cancelled = true; };
  }, []);
  // Stop any in-flight recording on unmount.
  useEffect(() => () => { try { mrRef.current?.stop(); } catch { /* noop */ } }, []);
  // Auto-grow the composer to fit its content (covers typing AND programmatic
  // changes like dictation/clear). Runs after React commits the new value, so
  // scrollHeight is accurate; grows up to a cap, then scrolls internally.
  useLayoutEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [draft]);

  // ── dictation: record on the phone, transcribe server-side, fill composer ──
  const toggleMic = useCallback(async () => {
    if (recording) { try { mrRef.current?.stop(); } catch { /* noop */ } return; }
    if (transcribing || conn !== "ready") return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data && e.data.size) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        if (blob.size < 1200) return; // too short / empty
        setTranscribing(true);
        try {
          const dataUrl = await blobToDataUrl(blob);
          const r = await fetchJSON<{ ok: boolean; transcript: string }>("/api/audio/transcribe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data_url: dataUrl, mime_type: blob.type }),
          });
          const text = (r.transcript || "").trim();
          if (text) {
            setDraft((d) => (d ? d.trimEnd() + " " : "") + text);
            requestAnimationFrame(() => taRef.current?.focus());
          }
        } catch { /* surface nothing for now; user can retry */ }
        finally { setTranscribing(false); }
      };
      mr.start();
      mrRef.current = mr;
      setRecording(true);
    } catch { /* mic permission denied / unavailable */ }
  }, [recording, transcribing, conn]);

  const onPickFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (!sidRef.current) return;
    for (const file of files) {
      const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
      const localId = uid();
      const reader = new FileReader();
      reader.onload = () => {
        const b64 = stripDataUrl(String(reader.result || ""));
        setPending((prev) => [...prev, { id: localId, name: file.name, kind: isPdf ? "pdf" : "image", thumb: isPdf ? undefined : String(reader.result || ""), status: "uploading" }]);
        const rid = isPdf
          ? sendRpc("pdf.attach", { session_id: sidRef.current, content_base64: b64, filename: file.name })
          : sendRpc("image.attach_bytes", { session_id: sidRef.current, data: b64, filename: file.name });
        attachRpc.current[rid] = localId;
      };
      reader.readAsDataURL(file);
    }
  };
  const removePending = (id: string) => setPending((p) => p.filter((a) => a.id !== id));

  const canSend = conn === "ready" && !running && (draft.trim().length > 0 || pending.some((p) => p.status === "ready"));
  const submit = useCallback(() => {
    const text = draft.trim();
    if (conn !== "ready" || running || !sidRef.current) return;
    const ready = pending.filter((p) => p.status === "ready");
    if (!text && ready.length === 0) return;
    push({ kind: "msg", id: uid(), role: "user", text, streaming: false, attachments: ready.map((a) => ({ name: a.name, kind: a.kind, thumb: a.thumb })) });
    setDraft(""); setPending([]); nearBottom.current = true; setRunning(true);
    sendRpc("prompt.submit", { session_id: sidRef.current, text: text || "(see attached)" });
    requestAnimationFrame(() => { if (taRef.current) taRef.current.style.height = "auto"; });
  }, [draft, conn, running, pending, sendRpc, push]);

  const stop = useCallback(() => { if (sidRef.current) sendRpc("session.interrupt", { session_id: sidRef.current }); }, [sendRpc]);
  const respondApproval = useCallback((item: ApprovalItem, choice: "approve" | "deny") => {
    sendRpc("approval.respond", { approval_id: item.approvalId, choice });
    patch(item.id, (it) => (it.kind === "approval" ? { ...it, resolved: choice } : it));
  }, [sendRpc, patch]);
  const newChat = useCallback(() => {
    setItems([]); setPending([]); curText.current = null; curReason.current = null; setRunning(false); sidRef.current = "";
    sendRpc("session.create", { source: "mobile-web", cols: 72, ...(profile ? { profile } : {}) });
  }, [sendRpc, profile]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } };
  const onInput = (e: React.FormEvent<HTMLTextAreaElement>) => setDraft(e.currentTarget.value);

  const statusLabel = useMemo(() => {
    if (conn === "ready") return running ? "thinking…" : model || "ready";
    if (conn === "connecting") return "connecting…";
    if (conn === "reconnecting") return "reconnecting…";
    return "disconnected";
  }, [conn, running, model]);
  const modelShort = useMemo(() => (model ? cap(model.split("-")[0]) : ""), [model]);

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-background text-foreground">
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex min-w-0 flex-col">
          <span className="text-display text-sm font-bold tracking-wide">Hermes</span>
          <span className="truncate text-xs text-text-secondary">
            <span className={"mr-1.5 inline-block h-2 w-2 rounded-full align-middle " + (conn === "ready" ? (running ? "bg-warning animate-pulse" : "bg-success") : "bg-text-tertiary")} />
            {statusLabel}
          </span>
        </div>
        <button onClick={newChat} className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs text-text-secondary active:scale-95"><Plus className="h-3.5 w-3.5" /> New</button>
      </header>

      <div ref={scrollRef} onScroll={onScroll} className="flex-1 overflow-y-auto overflow-x-hidden">
        {items.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center px-6 text-center">
            <div className="mb-3 text-5xl leading-none text-warning/80">☤</div>
            <div className="text-2xl font-medium tracking-tight text-foreground">
              {greeting()}{userName ? `, ${cap(userName)}` : ""}
            </div>
            {conn !== "ready" && <div className="mt-2 text-xs text-text-tertiary">{statusLabel}</div>}
          </div>
        ) : (
          <div className="space-y-3 px-3 py-4">
            {items.map((it) => <Row key={it.id} item={it} onApprove={respondApproval} />)}
          </div>
        )}
      </div>

      <div className="border-t border-border bg-background px-3 pt-2" style={{ paddingBottom: "max(0.5rem, env(safe-area-inset-bottom))" }}>
        {pending.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {pending.map((a) => (
              <div key={a.id} className={"relative flex items-center gap-1.5 rounded-lg border px-2 py-1 text-xs " + (a.status === "error" ? "border-destructive/50 text-destructive" : "border-border text-text-secondary")}>
                {a.kind === "image" && a.thumb ? <img src={a.thumb} alt="" className="h-8 w-8 rounded object-cover" /> : <FileText className="h-4 w-4" />}
                <span className="max-w-[8rem] truncate">{a.name}</span>
                {a.status === "uploading" && <span className="h-2 w-2 animate-pulse rounded-full bg-warning" />}
                <button onClick={() => removePending(a.id)} aria-label="Remove" className="ml-0.5 text-text-tertiary active:scale-90"><X className="h-3.5 w-3.5" /></button>
              </div>
            ))}
          </div>
        )}
        <div className="rounded-3xl border border-border bg-card px-3 py-2">
          <input ref={fileRef} type="file" accept="image/*,application/pdf" multiple hidden onChange={onPickFiles} />
          <textarea
            ref={taRef} value={draft} onInput={onInput} onKeyDown={onKeyDown} rows={1}
            placeholder={recording ? "Listening…" : transcribing ? "Transcribing…" : conn === "ready" ? "Message Hermes…" : statusLabel}
            className="max-h-[200px] w-full resize-none overflow-y-auto bg-transparent px-1 py-1 text-base text-foreground outline-none placeholder:text-text-tertiary"
          />
          <div className="mt-1 flex items-center gap-2">
            <button onClick={() => fileRef.current?.click()} disabled={conn !== "ready"} aria-label="Attach"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border text-text-secondary active:scale-95 disabled:opacity-30">
              <Plus className="h-5 w-5" />
            </button>
            {modelShort && <span className="truncate rounded-full bg-foreground/[0.06] px-2.5 py-1 text-xs text-text-secondary">{modelShort}</span>}
            <div className="flex-1" />
            <button onClick={toggleMic} disabled={(conn !== "ready" || running || transcribing) && !recording} aria-label="Dictate"
              className={"flex h-9 w-9 shrink-0 items-center justify-center rounded-full active:scale-95 disabled:opacity-30 " + (recording ? "bg-destructive text-text-on-accent animate-pulse" : "text-text-secondary")}>
              {transcribing ? <Loader2 className="h-5 w-5 animate-spin" /> : recording ? <Square className="h-4 w-4" /> : <Mic className="h-5 w-5" />}
            </button>
            {running ? (
              <button onClick={stop} aria-label="Stop" className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-warning text-text-on-accent active:scale-95"><Square className="h-4 w-4" /></button>
            ) : (
              <button onClick={submit} disabled={!canSend} aria-label="Send" className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-foreground text-background transition-opacity active:scale-95 disabled:opacity-30"><ArrowUp className="h-5 w-5" /></button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ item, onApprove }: { item: Item; onApprove: (it: ApprovalItem, choice: "approve" | "deny") => void }) {
  if (item.kind === "msg") return <MsgBubble item={item} />;
  if (item.kind === "reasoning") return <ReasoningBlock item={item} />;
  if (item.kind === "tool") return <ToolCard item={item} />;
  if (item.kind === "status") return <div className="text-center text-xs italic text-text-tertiary">{item.text}</div>;
  if (item.kind === "error") return <div className="rounded-xl border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">{item.text}</div>;
  if (item.kind === "approval") return <Approval item={item} onApprove={onApprove} />;
  return null;
}

function MsgBubble({ item }: { item: MsgItem }) {
  const mine = item.role === "user";
  if (mine) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl bg-foreground px-4 py-2.5 text-background">
          {item.attachments && item.attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {item.attachments.map((a, i) => a.kind === "image" && a.thumb ? <img key={i} src={a.thumb} alt="" className="h-20 w-20 rounded-lg object-cover" /> : <span key={i} className="flex items-center gap-1 rounded-lg bg-background/20 px-2 py-1 text-xs"><FileText className="h-3.5 w-3.5" />{a.name}</span>)}
            </div>
          )}
          {item.text && <p className="whitespace-pre-wrap break-words text-base">{item.text}</p>}
        </div>
      </div>
    );
  }
  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] rounded-2xl border border-border bg-card px-4 py-2.5 text-foreground">
        {item.text ? <Markdown content={item.text} streaming={item.streaming} /> : item.streaming ? <TypingDots /> : <span className="text-sm text-text-tertiary">(no response)</span>}
      </div>
    </div>
  );
}

function ReasoningBlock({ item }: { item: ReasoningItem }) {
  const [override, setOverride] = useState<boolean | null>(null);
  const open = override ?? item.streaming;
  return (
    <div className="rounded-xl border border-border/60 bg-card/30">
      <button onClick={() => setOverride(!open)} className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-text-tertiary">
        <span className="flex-1">{item.streaming ? "thinking…" : "thinking"}</span>
        {item.streaming && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-text-secondary" />}
        <ChevronDown className={"h-3.5 w-3.5 transition-transform " + (open ? "rotate-180" : "")} />
      </button>
      {open && <p className="max-h-48 overflow-auto whitespace-pre-wrap break-words border-t border-border/60 px-3 py-2 text-xs text-text-secondary">{item.text}</p>}
    </div>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex gap-1 py-1">
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-text-tertiary [animation-delay:-0.3s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-text-tertiary [animation-delay:-0.15s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-text-tertiary" />
    </span>
  );
}

function ToolCard({ item }: { item: ToolItem }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-border bg-card/50">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 px-3 py-2 text-left">
        <Wrench className="h-3.5 w-3.5 shrink-0 text-text-secondary" />
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-text-secondary">{item.name}{item.context ? ` · ${item.context}` : ""}</span>
        {!item.done ? <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-warning" /> : <span className="shrink-0 text-xs text-text-tertiary">{item.durationS != null ? `${item.durationS.toFixed(1)}s` : "done"}</span>}
        <ChevronDown className={"h-3.5 w-3.5 shrink-0 text-text-tertiary transition-transform " + (open ? "rotate-180" : "")} />
      </button>
      {open && (item.args || item.result) && (
        <div className="max-h-72 space-y-2 overflow-auto border-t border-border px-3 py-2">
          {item.args && <pre className="whitespace-pre-wrap break-words font-mono text-xs text-text-tertiary">{item.args}</pre>}
          {item.result && <pre className="whitespace-pre-wrap break-words font-mono text-xs text-text-secondary">{item.result}</pre>}
        </div>
      )}
    </div>
  );
}

function Approval({ item, onApprove }: { item: ApprovalItem; onApprove: (it: ApprovalItem, choice: "approve" | "deny") => void }) {
  return (
    <div className="rounded-xl border border-warning/50 bg-warning/10 px-3 py-3">
      <p className="text-sm text-foreground">{item.description}</p>
      {item.command && <pre className="mt-2 overflow-auto rounded-lg bg-background/60 px-2 py-1.5 font-mono text-xs text-text-secondary">{item.command}</pre>}
      {item.resolved ? (
        <p className="mt-2 text-xs text-text-tertiary">{item.resolved === "approve" ? "Approved" : "Denied"}</p>
      ) : (
        <div className="mt-3 flex gap-2">
          <button onClick={() => onApprove(item, "approve")} className="flex-1 rounded-lg bg-foreground py-2 text-sm font-medium text-background active:scale-95">Approve</button>
          <button onClick={() => onApprove(item, "deny")} className="flex-1 rounded-lg border border-border py-2 text-sm text-text-secondary active:scale-95">Deny</button>
        </div>
      )}
    </div>
  );
}
