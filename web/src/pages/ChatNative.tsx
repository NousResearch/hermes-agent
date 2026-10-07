import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";

import { ChatSessionList } from "@/components/ChatSessionList";
import { useProfileScope } from "@/contexts/useProfileScope";
import { api, type SessionMessage } from "@/lib/api";
import { GatewayClient } from "@/lib/gatewayClient";
import { cn } from "@/lib/utils";
import { ChatBar } from "@/chat/composer/ChatBar";

interface NativeMessage {
  id: string;
  role: SessionMessage["role"];
  text: string;
  pending?: boolean;
  error?: string;
}

function toNativeMessages(messages: SessionMessage[]): NativeMessage[] {
  return messages.map((m, i) => ({
    // eslint-disable-next-line no-underscore-dangle
    id: `hist-${i}-${m.timestamp ?? i}`,
    role: m.role,
    text: m.content ?? "",
  }));
}

function deltaText(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const text = (payload as { text?: unknown }).text;
  return typeof text === "string" ? text : "";
}

export default function ChatNative() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeSessionId = searchParams.get("resume");
  const { profile } = useProfileScope();
  const gw = useMemo(() => new GatewayClient(), []);
  const [messages, setMessages] = useState<NativeMessage[]>([]);
  const [gwSessionId, setGwSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scopeRef = useRef<string | null>(null);

  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      return;
    }
    const key = `${profile ?? ""}\0${activeSessionId}`;
    scopeRef.current = key;
    setError(null);
    api
      .getSessionMessages(activeSessionId)
      .then((res) => {
        if (scopeRef.current !== key) return;
        setMessages(toNativeMessages(res.messages));
      })
      .catch((e: Error) => {
        if (scopeRef.current !== key) return;
        setError(e.message || "failed to load messages");
      });
  }, [activeSessionId, profile]);

  useEffect(() => {
    let cancelled = false;
    const offDelta = gw.on("message.delta", (ev) => {
      const text = deltaText(ev.payload);
      if (!text) return;
      setMessages((prev) => {
        const last = prev.at(-1);
        if (!last || last.role !== "assistant" || !last.pending) return prev;
        return [...prev.slice(0, -1), { ...last, text: last.text + text }];
      });
    });
    const offComplete = gw.on("message.complete", () => {
      setSending(false);
      setMessages((prev) => {
        const last = prev.at(-1);
        if (!last || last.role !== "assistant") return prev;
        return [...prev.slice(0, -1), { ...last, pending: false }];
      });
    });
    gw.connect()
      .then(async () => {
        if (cancelled) return;
        if (activeSessionId) {
          try {
            await gw.request("session.resume", { session_id: activeSessionId });
            if (!cancelled) setGwSessionId(activeSessionId);
            return;
          } catch {
            /* fall through to create */
          }
        }
        const res = await gw.request<{ session_id: string }>("session.create", {
          source: "web",
          ...(profile ? { profile } : {}),
        });
        if (!cancelled) setGwSessionId(res.session_id);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message || "gateway connect failed");
      });
    return () => {
      cancelled = true;
      offDelta();
      offComplete();
    };
  }, [gw, activeSessionId, profile]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  const newChat = useCallback(() => {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete("resume");
        return next;
      },
      { replace: false },
    );
    setMessages([]);
    setGwSessionId(null);
    setError(null);
  }, [setSearchParams]);

  const send = useCallback(
    (text: string) => {
      if (!gwSessionId || sending) return;
      const now = Date.now();
      const user: NativeMessage = {
        id: `user-${now}`,
        role: "user",
        text,
      };
      const pending: NativeMessage = {
        id: `assistant-${now}`,
        role: "assistant",
        text: "",
        pending: true,
      };
      setMessages((prev) => [...prev, user, pending]);
      setSending(true);
      setError(null);
      gw.request("prompt.submit", { session_id: gwSessionId, text }).catch(
        (e: Error) => {
          setSending(false);
          setError(e.message || "send failed");
          setMessages((prev) => {
            const last = prev.at(-1);
            if (!last || last.role !== "assistant") return prev;
            return [
              ...prev.slice(0, -1),
              { ...last, pending: false, error: e.message },
            ];
          });
        },
      );
    },
    [gw, gwSessionId, sending],
  );

  return (
    <div className="flex min-h-0 flex-1 gap-4">
      <div className="hidden w-64 shrink-0 overflow-hidden border-r border-current/10 pr-2 lg:block">
        <ChatSessionList
          activeSessionId={activeSessionId}
          profile={profile}
          onNewChat={newChat}
        />
      </div>
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto">
          {error && (
            <div className="px-2 py-2 text-xs text-destructive">{error}</div>
          )}
          {messages.length === 0 && !error && (
            <div className="px-2 py-8 text-center text-sm text-text-secondary">
              New conversation — type below to start.
            </div>
          )}
          {messages.map((m) => (
            <div
              key={m.id}
              className={cn(
                "mx-1 my-2 max-w-[min(90%,44rem)] rounded-lg px-3 py-2 text-sm leading-6 wrap-anywhere",
                m.role === "user"
                  ? "ml-auto bg-primary/10 text-foreground"
                  : "bg-midground/5 text-foreground",
              )}
              data-role={m.role}
            >
              {m.text || (m.pending ? "…" : "")}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        <ChatBar onSend={send} disabled={!gwSessionId || sending} profile={profile} />
      </div>
    </div>
  );
}
