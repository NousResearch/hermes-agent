import { Button } from "@nous-research/ui/ui/components/button";
import { Markdown } from "@/components/Markdown";
import { HERMES_BASE_PATH, buildWsAuthParam, uploadChatFile } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Paperclip, Send, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type ChatRole = "user" | "assistant" | "system";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  status?: "streaming" | "complete" | "error";
};

type UploadAttachment = {
  name: string;
  path: string;
  mime_type: string;
  size: number;
  prompt_text: string;
  is_image: boolean;
};

type RpcPending = {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
};

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3 text-muted-foreground" aria-label="Hermes is working" role="status">
      <div className="flex items-center gap-1">
        <span className="h-2 w-2 animate-bounce rounded-full bg-current [animation-delay:-0.2s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-current [animation-delay:-0.1s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-current" />
      </div>
      <span className="text-xs font-medium uppercase tracking-wide">Hermes is working</span>
    </div>
  );
}

function buildNativeWsUrl(authParam: [string, string]): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const qs = new URLSearchParams({ [authParam[0]]: authParam[1] });
  return `${proto}//${window.location.host}${HERMES_BASE_PATH}/api/ws?${qs.toString()}`;
}

function makeId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function NativeChatPanel({ active }: { active: boolean }) {
  const wsRef = useRef<WebSocket | null>(null);
  const pendingRef = useRef<Map<string | number, RpcPending>>(new Map());
  const requestSeq = useRef(0);
  const sessionIdRef = useRef<string | null>(null);
  const currentAssistantIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "system",
      content:
        "Native chat beta is active. Drop or attach files here and Hermes will store them in the profile uploads folder, then reference them in your prompt.",
    },
  ]);
  const [composer, setComposer] = useState("");
  const [attachments, setAttachments] = useState<UploadAttachment[]>([]);
  const [connection, setConnection] = useState<"connecting" | "ready" | "error">("connecting");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);

  const rpc = useCallback((method: string, params: Record<string, unknown> = {}) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("Native chat socket is not connected"));
    }
    const id = ++requestSeq.current;
    const payload = { jsonrpc: "2.0", id, method, params };
    return new Promise((resolve, reject) => {
      pendingRef.current.set(id, { resolve, reject });
      ws.send(JSON.stringify(payload));
    });
  }, []);

  const appendMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const updateAssistant = useCallback((updater: (message: ChatMessage) => ChatMessage) => {
    const id = currentAssistantIdRef.current;
    if (!id) return;
    setMessages((prev) => prev.map((msg) => (msg.id === id ? updater(msg) : msg)));
  }, []);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let ws: WebSocket | null = null;

    async function connect() {
      setConnection("connecting");
      try {
        const auth = await buildWsAuthParam();
        if (cancelled) return;
        ws = new WebSocket(buildNativeWsUrl(auth));
        wsRef.current = ws;

        ws.onopen = () => {
          rpc("session.create", { cols: 96 })
            .then((result) => {
              const sessionId = (result as { session_id?: string }).session_id;
              if (sessionId) sessionIdRef.current = sessionId;
              setConnection("ready");
            })
            .catch((error) => {
              setConnection("error");
              appendMessage({ id: makeId("error"), role: "system", content: String(error), status: "error" });
            });
        };

        ws.onmessage = (event) => {
          let data: any;
          try {
            data = JSON.parse(event.data);
          } catch {
            return;
          }

          if (data.id !== undefined && pendingRef.current.has(data.id)) {
            const pending = pendingRef.current.get(data.id)!;
            pendingRef.current.delete(data.id);
            if (data.error) pending.reject(new Error(data.error.message || "RPC error"));
            else pending.resolve(data.result);
            return;
          }

          if (data.method !== "event") return;
          const type = data.params?.type;
          const payload = data.params?.payload || {};

          if (type === "message.start") {
            const id = makeId("assistant");
            currentAssistantIdRef.current = id;
            appendMessage({ id, role: "assistant", content: "", status: "streaming" });
            setBusy(true);
          } else if (type === "message.delta") {
            const text = String(payload.text || "");
            updateAssistant((msg) => ({ ...msg, content: msg.content + text, status: "streaming" }));
          } else if (type === "message.complete") {
            const text = String(payload.text || "");
            updateAssistant((msg) => ({
              ...msg,
              content: text || msg.content,
              status: payload.status === "error" ? "error" : "complete",
            }));
            currentAssistantIdRef.current = null;
            setBusy(false);
          } else if (type === "error") {
            appendMessage({
              id: makeId("error"),
              role: "system",
              content: String(payload.message || "Hermes reported an error."),
              status: "error",
            });
            setBusy(false);
          } else if (type === "status.update") {
            const text = String(payload.text || "").trim();
            if (text) appendMessage({ id: makeId("status"), role: "system", content: text });
          }
        };

        ws.onerror = () => setConnection("error");
        ws.onclose = () => {
          if (!cancelled) setConnection("error");
        };
      } catch (error) {
        setConnection("error");
        appendMessage({ id: makeId("error"), role: "system", content: String(error), status: "error" });
      }
    }

    void connect();

    return () => {
      cancelled = true;
      for (const pending of pendingRef.current.values()) {
        pending.reject(new Error("Native chat socket closed"));
      }
      pendingRef.current.clear();
      wsRef.current = null;
      ws?.close();
    };
  }, [active, appendMessage, rpc, updateAssistant]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  const attachmentText = useMemo(() => attachments.map((file) => file.prompt_text).join("\n"), [attachments]);

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const selected = Array.from(files);
    if (selected.length === 0) return;
    setUploading(true);
    try {
      const uploaded: UploadAttachment[] = [];
      for (const file of selected) {
        const result = await uploadChatFile(file);
        uploaded.push(result);
      }
      setAttachments((prev) => [...prev, ...uploaded]);
    } catch (error) {
      appendMessage({ id: makeId("upload-error"), role: "system", content: `Upload failed: ${error}`, status: "error" });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [appendMessage]);

  const send = useCallback(async () => {
    const text = composer.trim();
    if (!text && attachments.length === 0) return;
    const sessionId = sessionIdRef.current;
    if (!sessionId) return;

    const imageAttachments = attachments.filter((file) => file.is_image);
    for (const image of imageAttachments) {
      try {
        await rpc("image.attach", { session_id: sessionId, path: image.path });
      } catch {
        // Fall back to including the stored file path in the prompt below.
      }
    }

    const nonImageText = attachments
      .filter((file) => !file.is_image)
      .map((file) => file.prompt_text)
      .join("\n");
    const imageFallbackText = imageAttachments.map((file) => file.prompt_text).join("\n");
    const prompt = [nonImageText, imageFallbackText, text].filter(Boolean).join("\n\n");
    const visibleText = [attachmentText, text].filter(Boolean).join("\n\n");

    appendMessage({ id: makeId("user"), role: "user", content: visibleText });
    setComposer("");
    setAttachments([]);
    setBusy(true);

    try {
      await rpc("prompt.submit", { session_id: sessionId, text: prompt });
    } catch (error) {
      setBusy(false);
      appendMessage({ id: makeId("send-error"), role: "system", content: `Send failed: ${error}`, status: "error" });
    }
  }, [appendMessage, attachmentText, attachments, composer, rpc]);

  const showStandaloneTypingIndicator =
    busy && !messages.some((message) => message.role === "assistant" && message.status === "streaming");

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-border/70 bg-background/95 shadow-sm">
      <div className="flex items-center justify-between border-b border-border/70 px-4 py-3">
        <div>
          <div className="text-sm font-semibold">Native chat beta</div>
          <div className="text-xs text-muted-foreground">
            {connection === "ready" ? "Connected" : connection === "connecting" ? "Connecting…" : "Connection issue"}
          </div>
        </div>
        <div className="rounded-full border border-border px-2 py-1 text-[11px] uppercase tracking-wide text-muted-foreground">
          Files route to profile uploads
        </div>
      </div>

      <div
        className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-5"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          void handleFiles(event.dataTransfer.files);
        }}
      >
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn("flex", message.role === "user" ? "justify-end" : "justify-start")}
          >
            <div
              className={cn(
                "max-w-[min(760px,88%)] rounded-2xl px-4 py-3 text-sm shadow-sm",
                message.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : message.role === "system"
                    ? "border border-border bg-muted/45 text-muted-foreground"
                    : "border border-border bg-card text-card-foreground",
              )}
            >
              {message.role === "assistant" && message.status === "streaming" && !message.content ? (
                <TypingIndicator />
              ) : (
                <Markdown content={message.content} streaming={message.status === "streaming"} />
              )}
            </div>
          </div>
        ))}
        {showStandaloneTypingIndicator && (
          <div className="flex justify-start">
            <div className="max-w-[min(760px,88%)] rounded-2xl border border-border bg-card px-4 py-3 text-sm shadow-sm">
              <TypingIndicator />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 border-t border-border/70 px-4 py-2">
          {attachments.map((file) => (
            <span key={file.path} className="inline-flex max-w-full items-center gap-2 rounded-full border border-border bg-muted/60 px-3 py-1 text-xs">
              <Paperclip className="h-3 w-3 shrink-0" />
              <span className="truncate">{file.name}</span>
              <button
                type="button"
                className="text-muted-foreground hover:text-foreground"
                onClick={() => setAttachments((prev) => prev.filter((item) => item.path !== file.path))}
                aria-label={`Remove ${file.name}`}
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}

      <div className="border-t border-border/70 p-3">
        <div className="flex items-end gap-2 rounded-2xl border border-border bg-background px-3 py-2 focus-within:ring-2 focus-within:ring-ring/40">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(event) => event.target.files && void handleFiles(event.target.files)}
          />
          <Button
            ghost
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            aria-label="Attach files"
            title="Attach files"
          >
            <Paperclip className="h-4 w-4" />
          </Button>
          <textarea
            value={composer}
            onChange={(event) => setComposer(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
            placeholder="Ask Hermes anything, or attach/drop files here…"
            className="max-h-40 min-h-10 flex-1 resize-none bg-transparent py-2 text-sm outline-none placeholder:text-muted-foreground"
            rows={1}
            disabled={connection !== "ready"}
          />
          <Button onClick={() => void send()} disabled={connection !== "ready" || busy || uploading || (!composer.trim() && attachments.length === 0)}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
