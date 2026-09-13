/** Phone-first structured chat for an existing stored Hermes session.
 * History stays REST-backed; the live runtime is attached with session.resume
 * over the authenticated dashboard JSON-RPC socket. This deliberately does
 * not use the dashboard's xterm PTY surface. */
import { Button } from "@nous-research/ui/ui/components/button";
import { ArrowLeft, Send } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { api, type SessionMessage } from "@/lib/api";
import { ModelPickerDialog } from "@/components/ModelPickerDialog";
import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";

type LiveMessage = SessionMessage & { live?: boolean };
type ResumeResult = { session_id: string };
type TextPayload = { text?: string };

function messageText(message: SessionMessage) {
  return message.content ?? "";
}

export default function MobileSessionChatPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const storedId = params.get("session") ?? "";
  const profile = params.get("profile") ?? "";
  const [messages, setMessages] = useState<LiveMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [state, setState] = useState<ConnectionState>("idle");
  const [modelOpen, setModelOpen] = useState(false);
  const runtimeId = useRef<string | null>(null);
  const client = useRef<GatewayClient | null>(null);
  const bottom = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  useEffect(() => {
    if (!storedId) return;
    let cancelled = false;
    const gateway = new GatewayClient();
    client.current = gateway;
    const offState = gateway.onState(setState);
    const append = (text: string) => setMessages(previous => [...previous, { role: "assistant", content: text, live: true }]);
    const offStart = gateway.on<TextPayload>("message.start", () => append(""));
    const offDelta = gateway.on<TextPayload>("message.delta", event => {
      const text = event.payload?.text ?? "";
      if (!text) return;
      setMessages(previous => {
        const next = [...previous];
        const last = next.at(-1);
        if (last?.live && last.role === "assistant") last.content = `${last.content ?? ""}${text}`;
        else next.push({ role: "assistant", content: text, live: true });
        return next;
      });
    });
    void Promise.all([api.getSessionMessages(storedId, profile), gateway.connect()])
      .then(async ([history]) => {
        if (cancelled) return;
        setMessages(history.messages);
        const resumed = await gateway.request<ResumeResult>("session.resume", {
          session_id: storedId,
          source: "desktop",
          omit_messages: true,
          ...(profile ? { profile } : {}),
        });
        runtimeId.current = resumed.session_id;
      })
      .catch(e => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => { cancelled = true; offState(); offStart(); offDelta(); gateway.close(); client.current = null; };
  }, [storedId, profile]);

  const submit = async () => {
    const text = draft.trim();
    if (!text || !client.current || !runtimeId.current || state !== "open") return;
    setDraft("");
    setMessages(previous => [...previous, { role: "user", content: text, live: true }]);
    try { await client.current.request("prompt.submit", { session_id: runtimeId.current, text }, 1_800_000); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  if (!storedId) return <div className="p-4 text-sm text-destructive">Choose a session first.</div>;
  return <div className="relative mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col gap-3 pb-[env(safe-area-inset-bottom)]">
    <div className="flex items-center gap-2"><Button ghost size="icon" aria-label="Back to sessions" onClick={() => navigate("/sessions")}><ArrowLeft /></Button><div className="min-w-0 flex-1 truncate text-sm text-text-secondary">Desktop session</div><Button ghost className="min-h-11 px-3 text-sm" onClick={() => setModelOpen(true)}>Model</Button><span className="text-xs text-text-secondary">{state === "open" ? "live" : state}</span></div>
    {error && <div className="border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
    <main className="min-h-0 flex-1 space-y-3 overflow-y-auto px-1" aria-live="polite">
      {messages.map((message, index) => <div key={`${index}-${message.role}`} className={message.role === "user" ? "ml-8 rounded-2xl bg-primary px-3 py-2 text-primary-foreground" : "mr-4 rounded-2xl bg-muted px-3 py-2 text-sm whitespace-pre-wrap"}>{messageText(message) || (message.live ? "…" : "")}</div>)}
      <div ref={bottom} />
    </main>
    <form className="flex gap-2 border-t border-border pt-2" onSubmit={e => { e.preventDefault(); void submit(); }}><textarea value={draft} onChange={e => setDraft(e.target.value)} placeholder="Message Hermes" className="min-h-11 flex-1 resize-none rounded border border-border bg-background px-3 py-2 text-sm" disabled={state !== "open"} /><Button type="submit" size="icon" aria-label="Send message" disabled={!draft.trim() || state !== "open"}><Send /></Button></form>
    {modelOpen && <ModelPickerDialog loader={(options) => api.getModelOptions({ ...options, profile })} onApply={({ provider, model, confirmExpensiveModel }) => api.setModelAssignment({ scope: "main", provider, model, ...(confirmExpensiveModel ? { confirm_expensive_model: true } : {}) }, profile)} onClose={() => setModelOpen(false)} title="Choose model" />}
  </div>;
}
