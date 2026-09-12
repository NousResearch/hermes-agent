import { DRAFT_CONTROL_PREFIX, type DraftAttachRequest, type DraftIdentity, type DraftResult, type DraftState } from "@hermes/shared";
import { uploadChatFile } from "./chatFileTransfer";
import { parseDraftControl, sameDraft } from "./chatDraftProtocol";

export interface UploadContext {
  profile: string;
  scope: string;
  active: boolean;
}

export interface FileUpload {
  id: string;
  name: string;
  state: "queued" | "uploading" | "attaching" | "failed" | "unknown" | "stale";
  error?: string;
  canRetry: boolean;
}
interface Transfer extends Omit<FileUpload, "canRetry"> {
  file: File;
  context: UploadContext;
  socket: WebSocket;
  identity: DraftIdentity;
  path?: string;
  abort?: AbortController;
  timer?: ReturnType<typeof setTimeout>;
}

/** Browser bytes/control only. No terminal input, caret model or prompt submit. */
export class ChatDraftUploads {
  private context: UploadContext = { profile: "", scope: "", active: false };
  private socket: WebSocket | null = null;
  private draft: DraftState | null = null;
  private entries: Transfer[] = [];
  private nextId = 0;
  private snapshot = { uploads: [] as FileUpload[], available: false };
  private listeners = new Set<() => void>();

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };
  getSnapshot = () => this.snapshot;
  private available() {
    return this.context.active && !!this.draft?.available && this.socket?.readyState === WebSocket.OPEN;
  }
  private retryable(entry: Transfer) {
    return ["failed", "unknown"].includes(entry.state) && this.current(entry) && this.available();
  }
  private notify() {
    this.snapshot = {
      uploads: this.entries.map(entry => ({ id: entry.id, name: entry.name, state: entry.state, error: entry.error, canRetry: this.retryable(entry) })),
      available: this.available(),
    };
    this.listeners.forEach(listener => listener());
  }
  private current(entry: Transfer): boolean {
    return entry.state !== "stale" && this.context.active && entry.context.profile === this.context.profile &&
      entry.context.scope === this.context.scope && entry.socket === this.socket &&
      entry.socket.readyState === WebSocket.OPEN && !!this.draft && sameDraft(entry.identity, this.draft.identity);
  }
  private retireStale() {
    for (const entry of this.entries) {
      if (!this.current(entry)) this.fail(entry, "stale", "Draft changed.");
      else if (!this.draft?.available && entry.state !== "queued") this.fail(entry, "failed", "Composer unavailable.");
    }
  }
  setContext = (context: UploadContext) => {
    if (context.profile !== this.context.profile || context.scope !== this.context.scope) this.draft = null;
    this.context = { ...context };
    this.retireStale();
    this.notify();
  };
  bind = (socket: WebSocket | null) => {
    this.socket = socket;
    this.draft = null;
    this.retireStale();
    this.notify();
  };
  receive = (socket: WebSocket, data: string): boolean => {
    const message = parseDraftControl(data);
    if (message === undefined) return false;
    if (!message || socket !== this.socket) return true;
    if (message.type === "draft.state") {
      this.draft = message;
      this.retireStale();
    } else {
      this.acceptResult(message);
    }
    this.notify();
    return true;
  };
  private acceptResult(message: DraftResult) {
    const entry = this.entries.find(item => item.id === message.request_id);
    if (!entry?.path || !this.current(entry) || !sameDraft(entry.identity, message.identity)) return;
    clearTimeout(entry.timer);
    if (message.status === "attached") {
      this.entries = this.entries.filter(item => item !== entry);
      this.pump();
      return;
    }
    const errors = { failed: "Attachment failed.", unavailable: "Composer unavailable.", stale: "Draft changed." };
    this.fail(entry, message.status === "stale" ? "stale" : "failed", errors[message.status]);
    if (message.status === "stale") this.draft = null;
    if (message.status === "unavailable" && this.draft) this.draft = { ...this.draft, available: false };
    this.retireStale();
  }
  private requestId() {
    // randomUUID requires HTTPS; getRandomValues also works on remote HTTP.
    const random = Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, "0")).join("");
    return `${random}-${this.nextId++}`;
  }
  select = (files: File[]) => {
    if (!this.available() || !this.socket || !this.draft) return;
    for (const file of files) {
      this.entries.push({ id: this.requestId(), name: file.name, file, context: { ...this.context }, socket: this.socket, identity: { ...this.draft.identity }, state: "queued" });
    }
    this.pump();
    this.notify();
  };
  retry = (id: string) => {
    const entry = this.entries.find(item => item.id === id);
    if (!entry || !this.retryable(entry)) return;
    entry.state = "queued";
    entry.error = undefined;
    this.pump();
    this.notify();
  };
  remove = (id: string) => {
    const entry = this.entries.find(item => item.id === id);
    if (!entry) return;
    this.cancel(entry);
    this.entries = this.entries.filter(item => item !== entry);
    this.pump();
    this.notify();
  };
  dispose = () => {
    this.entries.forEach(entry => this.cancel(entry));
    this.entries = [];
    this.socket = null;
    this.draft = null;
    this.notify();
  };
  private pump() {
    // A failure/unknown outcome pauses later files until retry or dismissal,
    // so selection order survives errors as well as the happy path.
    const entry = this.entries.find(item => item.state !== "stale");
    if (!entry || entry.state !== "queued" || !this.available()) return;
    if (!this.current(entry)) { this.fail(entry, "stale", "Draft changed."); return; }
    if (entry.path) { this.attach(entry); return; }
    entry.state = "uploading";
    const abort = new AbortController();
    entry.abort = abort;
    void uploadChatFile(entry.file, entry.context.profile, abort.signal).then(result => {
      if (abort.signal.aborted) return;
      entry.path = result.path;
      this.attach(entry);
    }, () => {
      if (abort.signal.aborted) return;
      if (this.current(entry)) this.fail(entry, "failed", "Upload failed.");
      else this.fail(entry, "stale", "Draft changed.");
      this.notify();
    });
  }
  private cancel(entry: Transfer) {
    clearTimeout(entry.timer);
    entry.abort?.abort();
  }
  private fail(entry: Transfer, state: "failed" | "unknown" | "stale", error: string) {
    this.cancel(entry);
    entry.state = state;
    entry.error = error;
  }
  private attach(entry: Transfer) {
    if (!this.current(entry)) { this.fail(entry, "stale", "Draft changed."); this.notify(); return; }
    if (!this.available()) { this.fail(entry, "failed", "Composer unavailable."); this.notify(); return; }
    entry.state = "attaching";
    this.notify();
    const request: DraftAttachRequest = { type: "draft.attach", request_id: entry.id, expected: entry.identity, path: entry.path! };
    entry.timer = setTimeout(() => {
      if (this.current(entry)) this.fail(entry, "unknown", "Attachment not confirmed.");
      else this.fail(entry, "stale", "Draft changed.");
      this.notify();
    }, 30_000);
    try {
      entry.socket.send(DRAFT_CONTROL_PREFIX + JSON.stringify(request));
    } catch {
      this.fail(entry, "failed", "Connection failed.");
      this.notify();
    }
  }
}
