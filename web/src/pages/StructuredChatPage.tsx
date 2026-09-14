import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";
import {
  createStructuredChatState,
  isTranscriptPinnedToBottom,
  parseStructuredGatewayEvent,
  pinTranscriptToBottom,
  reduceStructuredChatEvent,
  replaceStructuredChatHistory,
  appendPendingUserMessage,
  STRUCTURED_HISTORY_POLL_MS,
  type StructuredChatState,
  type StructuredGatewayEvent,
  type StructuredHistoryMessage,
  type StructuredTimelineItem,
} from "@/lib/structured-chat";
import { buildWsUrl } from "@/lib/api";
import { ptyChatHref } from "@/lib/phone-structured-chat";

export interface StructuredGateway {
  connect(token?: string): Promise<void>;
  close(): void;
  onAny(handler: (event: StructuredGatewayEvent) => void): () => void;
  onState(handler: (state: ConnectionState) => void): () => void;
  request(method: string, params?: Record<string, unknown>): Promise<unknown>;
}

type ResumeResponse = {
  messages?: StructuredHistoryMessage[];
  messages_omitted?: boolean;
  hydrating?: boolean;
  running?: boolean;
  session_id?: string;
  status?: string;
  owner_id?: string;
  ownership_epoch?: number;
  read_only?: boolean;
};

type HistoryResponse = { messages?: StructuredHistoryMessage[] };

type StructuredChatPageProps = {
  clientFactory?: () => StructuredGateway;
  search?: string;
  followEvents?: (
    sessionIds: string[],
    onEvent: (event: StructuredGatewayEvent) => void,
  ) => () => void;
};

function defaultFollowEvents(
  sessionIds: string[],
  onEvent: (event: StructuredGatewayEvent) => void,
): () => void {
  if (typeof WebSocket === "undefined") return () => undefined;
  const sockets: WebSocket[] = [];
  let cancelled = false;
  for (const sessionId of [...new Set(sessionIds.filter(Boolean))]) {
    void (async () => {
      try {
        const url = await buildWsUrl("/api/events", { session: sessionId });
        if (cancelled) return;
        const socket = new WebSocket(url);
        sockets.push(socket);
        socket.onmessage = (message) => {
          const event = parseStructuredGatewayEvent(String(message.data ?? ""));
          if (event) onEvent(event);
        };
      } catch {
        // Follow is best-effort; history already loaded. ChatSidebar same policy.
      }
    })();
  }
  return () => {
    cancelled = true;
    for (const socket of sockets) socket.close();
  };
}

function TimelineItem({
  item,
  onApproval,
  actionsDisabled,
}: {
  item: StructuredTimelineItem;
  onApproval: (requestId: string, choice: "once" | "deny") => void;
  actionsDisabled: boolean;
}) {
  if (item.kind === "tool") {
    return (
      <details key={item.id} open={item.status === "running" || item.status === "error"} className="rounded border border-current/20 p-3">
        <summary className="cursor-pointer font-medium">{item.title} · {item.status}</summary>
        {item.text && <pre className="mt-2 whitespace-pre-wrap break-words text-xs">{item.text}</pre>}
      </details>
    );
  }
  if (item.kind === "subagent") {
    return (
      <article key={item.id} className="ml-4 rounded border border-current/20 p-3" aria-label={`Subagent ${item.title}`}>
        <strong>{item.title}</strong> <span>{item.role}</span> · <span>{item.status}</span>
        {item.text && <p className="whitespace-pre-wrap">{item.text}</p>}
      </article>
    );
  }
  if (item.kind === "interaction") {
    const isApproval = item.title === "approval";
    return (
      <article key={item.id} className="rounded border border-amber-500/50 p-3" data-request-id={item.id}>
        <strong>{item.title}</strong>
        <p className="whitespace-pre-wrap">{item.text}</p>
        <span>{item.status}</span>
        {isApproval && item.status === "pending" && (
          <div className="mt-2 flex gap-2">
            <button type="button" disabled={actionsDisabled} onClick={() => onApproval(item.id, "once")}>Erlauben</button>
            <button type="button" disabled={actionsDisabled} onClick={() => onApproval(item.id, "deny")}>Ablehnen</button>
          </div>
        )}
      </article>
    );
  }
  return (
    <article key={item.id} className="rounded border border-current/10 p-3" data-kind={item.kind}>
      {item.title && <strong>{item.title}</strong>}
      <p className="whitespace-pre-wrap break-words">{item.text}</p>
    </article>
  );
}

function describeGatewayError(cause: unknown): string {
  const message = cause instanceof Error ? cause.message : String(cause);
  return message.includes("SESSION_NOT_OWNED")
    ? "Eine andere Laufzeit besitzt diese Session. Der Text bleibt erhalten; es wurde nichts erneut gesendet."
    : message;
}

function browserOwnerId(): string {
  const key = "hermes-structured-owner-id";
  try {
    const existing = sessionStorage.getItem(key);
    if (existing) return existing;
    const created = crypto.randomUUID();
    sessionStorage.setItem(key, created);
    return created;
  } catch {
    return crypto.randomUUID();
  }
}

function createGatewayClient(): StructuredGateway {
  return new GatewayClient();
}

export default function StructuredChatPage({
  clientFactory = createGatewayClient,
  search = window.location.search,
  followEvents = defaultFollowEvents,
}: StructuredChatPageProps) {
  const params = useMemo(() => new URLSearchParams(search), [search]);
  const durableSessionId = params.get("resume")?.trim() ?? "";
  const profile = params.get("profile")?.trim() ?? "";
  const clientRef = useRef<StructuredGateway | null>(null);
  const runtimeIdRef = useRef("");
  const bufferedEventsRef = useRef<StructuredGatewayEvent[]>([]);
  const [timeline, setTimeline] = useState<StructuredChatState>(() => createStructuredChatState(durableSessionId));
  const [runtimeId, setRuntimeId] = useState("");
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [composer, setComposer] = useState("");
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [readOnly, setReadOnly] = useState(!durableSessionId);
  const [ownerId] = useState(browserOwnerId);
  const [ownershipEpoch, setOwnershipEpoch] = useState(0);
  const [error, setError] = useState(
    durableSessionId ? "" : "Eine explizite Session-ID in ?resume= ist erforderlich.",
  );
  const transcriptRef = useRef<HTMLElement | null>(null);
  const pinnedToBottomRef = useRef(true);

  useEffect(() => {
    if (!durableSessionId) return;

    let cancelled = false;
    const client = clientFactory();
    clientRef.current = client;
    const offState = client.onState(setConnection);
    const offEvents = client.onAny((event) => {
      if (event.type === "session.revoked") {
        setReadOnly(true);
        setError("Schreibrecht entzogen. Der Composer ist gesperrt.");
      }
      if (!runtimeIdRef.current) {
        bufferedEventsRef.current.push(event);
        return;
      }
      setTimeline((current) => reduceStructuredChatEvent(current, event));
    });

    const onFollowEvent = (event: StructuredGatewayEvent) => {
      if (!runtimeIdRef.current) {
        bufferedEventsRef.current.push(event);
        return;
      }
      setTimeline((current) => reduceStructuredChatEvent(current, event));
    };
    let stopFollow = followEvents([durableSessionId], onFollowEvent);

    void (async () => {
      try {
        await client.connect();
        const resumeParams: Record<string, unknown> = { session_id: durableSessionId, owner_id: ownerId };
        if (profile) resumeParams.profile = profile;
        const resumed = await client.request("session.resume", resumeParams) as ResumeResponse;
        const runtimeId = resumed.session_id;
        if (!runtimeId) throw new Error("Hermes lieferte keine Runtime-Session-ID.");
        runtimeIdRef.current = runtimeId;
        const needsHistory = Boolean(resumed.messages_omitted || resumed.hydrating || !resumed.messages);
        const history = needsHistory
          ? await client.request("session.history", { session_id: runtimeId }) as HistoryResponse
          : { messages: resumed.messages };
        if (cancelled) return;
        setRuntimeId(runtimeId);
        setOwnershipEpoch(typeof resumed.ownership_epoch === "number" ? resumed.ownership_epoch : 0);
        let next = replaceStructuredChatHistory(
          createStructuredChatState(runtimeId, [durableSessionId]),
          history.messages ?? resumed.messages ?? [],
        );
        for (const event of bufferedEventsRef.current) next = reduceStructuredChatEvent(next, event);
        bufferedEventsRef.current = [];
        setTimeline(next);
        setReadOnly(Boolean(resumed.read_only));
        stopFollow();
        stopFollow = followEvents([durableSessionId, runtimeId], onFollowEvent);
        const poll = window.setInterval(() => {
          const liveClient = clientRef.current;
          const liveRuntime = runtimeIdRef.current;
          if (!liveClient || !liveRuntime) return;
          void liveClient.request("session.history", { session_id: liveRuntime }).then((raw) => {
            if (cancelled) return;
            const messages = (raw as HistoryResponse).messages ?? [];
            if (!messages.length) return;
            setTimeline((current) => {
              const next = replaceStructuredChatHistory(current, messages);
              return { ...next, lastSeq: current.lastSeq };
            });
          }).catch(() => undefined);
        }, STRUCTURED_HISTORY_POLL_MS);
        const previousStop = stopFollow;
        stopFollow = () => {
          window.clearInterval(poll);
          previousStop();
        };
      } catch (cause) {
        if (cancelled) return;
        setReadOnly(true);
        setError(describeGatewayError(cause));
      }
    })();

    return () => {
      cancelled = true;
      stopFollow();
      offEvents();
      offState();
      client.close();
      clientRef.current = null;
      runtimeIdRef.current = "";
    };
  }, [clientFactory, durableSessionId, followEvents, ownerId, profile]);

  useEffect(() => {
    const node = transcriptRef.current;
    if (!node || !pinnedToBottomRef.current) return;
    pinTranscriptToBottom(node);
  }, [timeline.items]);

  const submit = useCallback(async () => {
    const value = (composerRef.current?.value ?? composer).trim();
    const runtimeId = runtimeIdRef.current;
    if (!value) return;
    if (readOnly) {
      setError("Diese Session schreibt noch in der TUI. Übernimm sie, sonst kommt die Nachricht nicht an.");
      return;
    }
    if (connection !== "open" || !runtimeId || !clientRef.current) {
      setError("Keine Verbindung zum Gateway. Die Nachricht wurde nicht gesendet.");
      return;
    }
    if (submitting) return;
    setSubmitting(true);
    setError("");
    try {
      await clientRef.current.request("prompt.submit", {
        session_id: runtimeId,
        text: value,
        owner_id: ownerId,
        ownership_epoch: ownershipEpoch,
      });
      setComposer("");
      if (composerRef.current) composerRef.current.value = "";
      setTimeline((current) => appendPendingUserMessage(current, value));
    } catch (cause) {
      setReadOnly(true);
      setError(describeGatewayError(cause));
    } finally {
      setSubmitting(false);
    }
  }, [composer, connection, ownerId, ownershipEpoch, readOnly, submitting]);

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submit();
  };
  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void submit();
  };
  const takeover = useCallback(async () => {
    const runtimeId = runtimeIdRef.current;
    if (!runtimeId || !clientRef.current || connection !== "open") return;
    try {
      const result = await clientRef.current.request("session.takeover", {
        session_id: runtimeId,
        owner_id: ownerId,
        ownership_epoch: ownershipEpoch,
        confirmed: true,
      }) as ResumeResponse;
      setOwnershipEpoch(typeof result.ownership_epoch === "number" ? result.ownership_epoch : ownershipEpoch + 1);
      setReadOnly(false);
      setError("");
    } catch (cause) {
      setReadOnly(true);
      setError(describeGatewayError(cause));
    }
  }, [connection, ownerId, ownershipEpoch]);
  useEffect(() => {
    const viewport = window.visualViewport;
    if (!viewport) return;
    const apply = () => {
      const inset = Math.max(0, window.innerHeight - viewport.height - viewport.offsetTop);
      document.documentElement.style.setProperty("--structured-vv-inset", `${inset}px`);
    };
    viewport.addEventListener("resize", apply);
    viewport.addEventListener("scroll", apply);
    apply();
    return () => {
      viewport.removeEventListener("resize", apply);
      viewport.removeEventListener("scroll", apply);
      document.documentElement.style.removeProperty("--structured-vv-inset");
    };
  }, []);
  const respondToApproval = useCallback(async (requestId: string, choice: "once" | "deny") => {
    const runtimeId = runtimeIdRef.current;
    if (!runtimeId || !clientRef.current || readOnly || connection !== "open") return;
    try {
      await clientRef.current.request("approval.respond", {
        session_id: runtimeId,
        request_id: requestId,
        choice,
        owner_id: ownerId,
        ownership_epoch: ownershipEpoch,
      });
      setTimeline((current) => ({
        ...current,
        items: current.items.map((item) => item.id === requestId ? { ...item, status: "beantwortet" } : item),
      }));
    } catch (cause) {
      setError(describeGatewayError(cause));
    }
  }, [connection, ownerId, ownershipEpoch, readOnly]);
  const unavailable = readOnly || submitting || connection !== "open" || !runtimeId;

  return (
    <main className="flex min-h-0 flex-1 flex-col overflow-hidden" aria-label="Strukturierter Hermes-Chat">
      <header className="flex flex-wrap items-center gap-2 border-b border-current/20 py-3">
        <h1 className="font-semibold">Hermes Chat</h1>
        <code className="text-xs">{durableSessionId || "keine Session"}</code>
        <span role="status">{readOnly ? "Nur Lesen — TUI schreibt. Übernehmen, sonst kommt nichts an." : connection === "open" ? "Schreibend" : "Verbinden…"}</span>
        {readOnly && runtimeId && connection === "open" && (
          <button type="button" onClick={() => void takeover()}>Session übernehmen</button>
        )}
      </header>

      {error && <div role="alert" className="border-b border-red-500/40 p-3 text-red-500">{error}</div>}

      <section
        ref={transcriptRef}
        className="min-h-0 flex-1 space-y-3 overflow-y-auto py-4"
        aria-label="Chatverlauf"
        aria-live="polite"
        onScroll={(event) => {
          pinnedToBottomRef.current = isTranscriptPinnedToBottom(event.currentTarget);
        }}
      >
        {timeline.items.map((item) => (
          <TimelineItem key={item.id} item={item} onApproval={respondToApproval} actionsDisabled={unavailable} />
        ))}
      </section>

      <details className="border-t border-current/20 py-2">
        <summary>Diagnose-Terminal</summary>
        {readOnly || !durableSessionId ? (
          <p className="text-sm text-text-secondary">Geschlossen. Schreibbesitz muss zuerst eindeutig hier liegen; sonst entsteht ein zweiter Schreiber.</p>
        ) : (
          <p className="text-sm">
            <a href={ptyChatHref(durableSessionId, profile)}>PTY derselben Session öffnen</a>
            {" "}— der strukturierte Client wird beim Verlassen beendet.
          </p>
        )}
      </details>

      <form onSubmit={onSubmit} className="border-t border-current/20 py-3" style={{ paddingBottom: "calc(0.75rem + var(--structured-vv-inset, 0px))" }}>
        <label className="sr-only" htmlFor="structured-chat-composer">Nachricht</label>
        <textarea
          id="structured-chat-composer"
          ref={composerRef}
          value={composer}
          onChange={(event) => setComposer(event.currentTarget.value)}
          onKeyDown={onComposerKeyDown}
          disabled={unavailable}
          rows={3}
          enterKeyHint="send"
          autoCorrect="on"
          autoCapitalize="sentences"
          autoComplete="on"
          spellCheck
          inputMode="text"
          className="w-full resize-y rounded border border-current/30 bg-transparent p-3"
        />
        <button type="submit" disabled={unavailable || !composer.trim()} className="mt-2 rounded border border-current/30 px-4 py-2">
          {submitting ? "Senden…" : "Senden"}
        </button>
      </form>
    </main>
  );
}
