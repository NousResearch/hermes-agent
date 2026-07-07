import { useEffect, useState, type CSSProperties } from "react";
import type { SessionInfo, SessionMessage, SessionMessagesResponse } from "@/lib/api";
import { del, get, patch, post } from "../api";
import { useMiniApp } from "../context";

function relativeAgo(epochSeconds: number): string {
  const s = Date.now() / 1000 - epochSeconds;
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function bubbleStyle(role: SessionMessage["role"]): CSSProperties {
  switch (role) {
    case "user":
      return {
        alignSelf: "flex-end",
        maxWidth: "80%",
        background: "var(--card2)",
        border: "1px solid var(--line)",
        borderRadius: "15px 15px 4px 15px",
        padding: "9px 13px",
        fontSize: 13,
        lineHeight: 1.45,
        color: "var(--mid)",
      };
    case "tool":
      return {
        alignSelf: "flex-start",
        maxWidth: "85%",
        border: "1px dashed var(--line2)",
        borderRadius: 10,
        padding: "7px 11px",
        fontFamily: "var(--mono)",
        fontSize: 11,
        lineHeight: 1.5,
        color: "var(--t3)",
      };
    case "system":
      return {
        alignSelf: "center",
        fontFamily: "var(--mono)",
        fontSize: 10.5,
        color: "var(--t3)",
        padding: "2px 6px",
      };
    default:
      return {
        alignSelf: "flex-start",
        maxWidth: "85%",
        background: "var(--card)",
        border: "1px solid var(--line)",
        borderRadius: "15px 15px 15px 4px",
        padding: "9px 13px",
        fontSize: 13,
        lineHeight: 1.5,
        color: "var(--t2)",
      };
  }
}

export function SessionDetailPane({
  id,
  onNotFound,
  onBack,
}: {
  id: string;
  onNotFound: () => void;
  onBack: () => void;
}) {
  const { isAdmin, showToast, askConfirm } = useMiniApp();
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [messages, setMessages] = useState<SessionMessage[] | null>(null);

  // No reset-to-null at the top of this effect: MiniApp.tsx mounts this
  // component with `key={detail.id}`, so a different session id is a fresh
  // instance (state already starts at null) rather than the same instance
  // needing to clear stale session/messages state itself.
  useEffect(() => {
    get<SessionInfo>(`/api/sessions/${id}`)
      .then(setSession)
      .catch(() => onNotFound());
    get<SessionMessagesResponse>(`/api/sessions/${id}/messages`)
      .then((r) => setMessages(r.messages))
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  if (!session) return null;

  const archive = () =>
    askConfirm({
      title: "Archive this session?",
      body: "The session moves out of the active list. Message history is kept and it can be restored from the desktop dashboard.",
      label: "Archive session",
      destructive: false,
      run: async () => {
        try {
          await patch(`/api/sessions/${id}`, { archived: true });
          showToast("Session archived");
          onBack();
        } catch {
          showToast("Couldn't archive session");
        }
      },
    });

  const remove = () =>
    askConfirm({
      title: "Delete this session?",
      body: "Message history is permanently removed for everyone. This cannot be undone.",
      label: "Delete session",
      destructive: true,
      run: async () => {
        try {
          await del(`/api/sessions/${id}`);
          showToast("Session deleted");
          onBack();
        } catch {
          showToast("Couldn't delete session");
        }
      },
    });

  const resume = () =>
    askConfirm({
      title: "Resume this session?",
      body: "Makes this the active session for its Telegram chat, within a few seconds — the chat continues from here instead of wherever it currently is.",
      label: "Resume",
      destructive: false,
      run: async () => {
        try {
          await post(`/api/sessions/${id}/resume`);
          showToast("Resuming — takes effect within a few seconds");
        } catch {
          showToast("Couldn't resume session");
        }
      },
    });

  return (
    <div style={{ padding: "16px 14px 24px", display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, padding: "12px 14px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              fontSize: 14,
              fontWeight: 650,
              color: "var(--mid)",
              flex: 1,
              minWidth: 0,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {session.title || "Untitled session"}
          </span>
          {session.source && (
            <span
              style={{
                fontFamily: "var(--mono)",
                fontSize: 10,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                border: "1px solid var(--line)",
                borderRadius: 6,
                padding: "2px 6px",
                color: "var(--t3)",
                whiteSpace: "nowrap",
                flexShrink: 0,
              }}
            >
              {session.source}
            </span>
          )}
        </div>
        <div style={{ display: "flex", gap: 12, marginTop: 8, fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--t3)", flexWrap: "wrap" }}>
          {session.model && <span>{session.model}</span>}
          <span>{session.message_count} msgs</span>
          <span>
            {Math.round(session.input_tokens / 1000)}k in · {Math.round(session.output_tokens / 1000)}k out
          </span>
          <span style={{ marginLeft: "auto" }}>{relativeAgo(session.last_active)}</span>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {(messages ?? []).map((m, i) => (
          <div key={i} style={bubbleStyle(m.role)}>
            {m.content ?? (m.tool_name ? `${m.tool_name}(...)` : "")}
          </div>
        ))}
      </div>

      {isAdmin && (
        <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
          {session.source === "telegram" && (
            <button
              onClick={resume}
              style={{
                flex: 1,
                padding: 11,
                borderRadius: 11,
                border: "1px solid color-mix(in srgb, var(--accent) 45%, transparent)",
                background: "transparent",
                fontSize: 13,
                fontWeight: 600,
                color: "var(--accent)",
                cursor: "pointer",
              }}
            >
              Resume
            </button>
          )}
          <button
            onClick={archive}
            style={{
              flex: 1,
              padding: 11,
              borderRadius: 11,
              border: "1px solid var(--line2)",
              background: "transparent",
              fontSize: 13,
              fontWeight: 600,
              color: "var(--t2)",
              cursor: "pointer",
            }}
          >
            Archive
          </button>
          <button
            onClick={remove}
            style={{
              flex: 1,
              padding: 11,
              borderRadius: 11,
              border: "1px solid color-mix(in srgb, var(--destr) 45%, transparent)",
              background: "transparent",
              fontSize: 13,
              fontWeight: 600,
              color: "var(--destr)",
              cursor: "pointer",
            }}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  );
}
