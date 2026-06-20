/**
 * useCopilotSessions — multi-session (tabbed) state + gateway wiring for the
 * 爱马仕 Copilot page.
 *
 * One GatewayClient drives N sessions: each open tab is a gateway session
 * (`session.create`), and stream events fan out tagged with `session_id`, so a
 * background tab keeps receiving its turn while you read another. The active
 * tab is just which session's messages we render.
 *
 * History is server-backed via the gateway's session store:
 *   session.list     → past conversations (id/title/preview/started_at/count)
 *   session.resume   → re-attach a stored session so it's live again
 *   session.history  → its messages (to seed the reopened tab)
 *   session.title    → rename     session.delete → drop
 * pin / archive / search are client-side (localStorage + filtering) since the
 * gateway has no such concept.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";

export interface CopilotMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  reasoning?: string;
  status?: string; // live status label while streaming (tool / thinking)
  error?: string;
  done?: boolean;
}

export interface CopilotTab {
  /** gateway session_id */
  id: string;
  title: string;
  messages: CopilotMessage[];
  /** true while an assistant turn is streaming for this session */
  busy: boolean;
}

export interface HistorySession {
  id: string;
  title: string;
  preview: string;
  started_at: number;
  message_count: number;
  source: string;
}

const TABS_STORAGE_KEY = "hermes.copilot.openTabs.v1";

const uid = () =>
  globalThis.crypto?.randomUUID?.() ??
  `m-${Math.random().toString(36).slice(2)}`;

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  if (Array.isArray(value)) return value.map(asText).join("");
  if (typeof value === "object") {
    const row = value as Record<string, unknown>;
    if (typeof row.text === "string") return row.text;
    if (typeof row.output_text === "string") return row.output_text;
  }
  return "";
}

function readPersistedTabs(): { id: string; title: string }[] {
  try {
    const raw = localStorage.getItem(TABS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (t): t is { id: string; title: string } =>
          t && typeof t.id === "string",
      )
      .map((t) => ({ id: t.id, title: typeof t.title === "string" ? t.title : "" }));
  } catch {
    return [];
  }
}

/** Coerce a gateway history message into our local bubble shape. */
function historyToMessages(raw: unknown): CopilotMessage[] {
  if (!Array.isArray(raw)) return [];
  const out: CopilotMessage[] = [];
  for (const m of raw) {
    if (!m || typeof m !== "object") continue;
    const row = m as Record<string, unknown>;
    const role = row.role === "user" ? "user" : "assistant";
    const text = asText(row.content ?? row.text);
    const reasoning = asText(row.reasoning);
    if (!text && !reasoning) continue;
    out.push({
      id: uid(),
      role,
      text,
      reasoning: reasoning || undefined,
      done: true,
    });
  }
  return out;
}

export interface UseCopilotSessions {
  state: ConnectionState;
  model: string;
  banner: string | null;
  tabs: CopilotTab[];
  activeId: string | null;
  activeTab: CopilotTab | null;
  send: (text: string) => void;
  stop: () => void;
  newTab: () => void;
  closeTab: (id: string) => void;
  switchTab: (id: string) => void;
  /** switch the active session's model in place (e.g. 性能 ↔ 极致); pass the
   *  provider so the gateway routes to the right downlink (not openrouter) */
  switchModel: (model: string, provider?: string) => Promise<void>;
  /** resume a stored session into a (new or existing) tab and focus it */
  openHistory: (session: HistorySession) => void;
  listHistory: () => Promise<HistorySession[]>;
  renameSession: (id: string, title: string) => Promise<void>;
  deleteSession: (id: string) => Promise<void>;
  reconnect: () => void;
}

export function useCopilotSessions(): UseCopilotSessions {
  // `version` bumps on reconnect; deriving the client from it tears down the
  // old socket and builds a fresh one (mirrors the original CopilotPage).
  const [version, setVersion] = useState(0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const gw = useMemo(() => new GatewayClient(), [version]);

  const [state, setState] = useState<ConnectionState>("idle");
  const [model, setModel] = useState("");
  const [banner, setBanner] = useState<string | null>(null);
  const [tabs, setTabs] = useState<CopilotTab[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  // id of the assistant bubble currently streaming, per session_id. Lives in a
  // ref so event handlers (bound once per connection) read the latest without
  // re-subscribing on every keystroke.
  const streamingRef = useRef<Record<string, string | null>>({});
  const activeIdRef = useRef<string | null>(null);
  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  const knownSession = useCallback(
    (sid?: string) => !!sid && sid in streamingRef.current,
    [],
  );

  const patchMessages = useCallback(
    (sid: string, fn: (msgs: CopilotMessage[]) => CopilotMessage[]) => {
      setTabs((prev) =>
        prev.map((t) => (t.id === sid ? { ...t, messages: fn(t.messages) } : t)),
      );
    },
    [],
  );

  const patchStreaming = useCallback(
    (sid: string, fn: (m: CopilotMessage) => CopilotMessage) => {
      const streamId = streamingRef.current[sid];
      if (!streamId) return;
      patchMessages(sid, (msgs) =>
        msgs.map((m) => (m.id === streamId ? fn(m) : m)),
      );
    },
    [patchMessages],
  );

  const setBusy = useCallback((sid: string, busy: boolean) => {
    setTabs((prev) => prev.map((t) => (t.id === sid ? { ...t, busy } : t)));
  }, []);

  const persistTabs = useCallback((next: CopilotTab[]) => {
    try {
      localStorage.setItem(
        TABS_STORAGE_KEY,
        JSON.stringify(next.map((t) => ({ id: t.id, title: t.title }))),
      );
    } catch {
      /* best-effort */
    }
  }, []);

  // Keep localStorage in sync whenever the open-tab set/titles change.
  useEffect(() => {
    persistTabs(tabs);
  }, [tabs, persistTabs]);

  // --- gateway lifecycle + stream handlers (bound once per connection) ------
  useEffect(() => {
    let cancelled = false;
    const offs: Array<() => void> = [];
    offs.push(gw.onState(setState));

    offs.push(
      gw.on("session.info", (ev) => {
        const p = ev.payload as { model?: string } | undefined;
        if (typeof p?.model === "string" && p.model) setModel(p.model);
      }),
    );

    offs.push(
      gw.on("message.start", (ev) => {
        const sid = ev.session_id;
        if (!knownSession(sid) || !sid) return;
        if (!streamingRef.current[sid]) {
          const id = uid();
          streamingRef.current[sid] = id;
          patchMessages(sid, (msgs) => [
            ...msgs,
            { id, role: "assistant", text: "", status: "thinking" },
          ]);
          setBusy(sid, true);
        }
      }),
    );

    offs.push(
      gw.on("message.delta", (ev) => {
        const sid = ev.session_id;
        if (!knownSession(sid) || !sid) return;
        const text = asText((ev.payload as { text?: unknown })?.text);
        if (!text) return;
        patchStreaming(sid, (m) => ({
          ...m,
          text: m.text + text,
          status: undefined,
        }));
      }),
    );

    const onReasoning =
      (replace: boolean) => (ev: { session_id?: string; payload?: unknown }) => {
        const sid = ev.session_id;
        if (!knownSession(sid) || !sid) return;
        const text = asText((ev.payload as { text?: unknown })?.text);
        if (!text) return;
        patchStreaming(sid, (m) => ({
          ...m,
          reasoning: replace ? text : (m.reasoning ?? "") + text,
        }));
      };
    offs.push(gw.on("reasoning.delta", onReasoning(false)));
    offs.push(gw.on("reasoning.available", onReasoning(true)));

    const onTool =
      (running: boolean) => (ev: { session_id?: string; payload?: unknown }) => {
        const sid = ev.session_id;
        if (!knownSession(sid) || !sid) return;
        const p = ev.payload as { name?: string } | undefined;
        patchStreaming(sid, (m) => ({
          ...m,
          status: running ? `running_tool:${p?.name ?? ""}` : "thinking",
        }));
      };
    offs.push(gw.on("tool.start", onTool(true)));
    offs.push(gw.on("tool.progress", onTool(true)));
    offs.push(gw.on("tool.complete", onTool(false)));

    offs.push(
      gw.on("message.complete", (ev) => {
        const sid = ev.session_id;
        if (!knownSession(sid) || !sid) return;
        const p = ev.payload as { text?: unknown; rendered?: unknown } | undefined;
        const finalText = asText(p?.text) || asText(p?.rendered);
        patchStreaming(sid, (m) => ({
          ...m,
          text: finalText || m.text,
          status: undefined,
          done: true,
        }));
        streamingRef.current[sid] = null;
        setBusy(sid, false);
      }),
    );

    offs.push(
      gw.on("error", (ev) => {
        const sid = ev.session_id;
        if (!knownSession(sid) || !sid) return;
        const message =
          (ev.payload as { message?: string } | undefined)?.message ||
          "出错了,请重试";
        if (streamingRef.current[sid]) {
          patchStreaming(sid, (m) => ({ ...m, status: undefined, error: message }));
          streamingRef.current[sid] = null;
        } else {
          patchMessages(sid, (msgs) => [
            ...msgs,
            { id: uid(), role: "assistant", text: "", error: message },
          ]);
        }
        setBusy(sid, false);
      }),
    );

    // Connect, then restore persisted tabs (resume each) or open a fresh one.
    gw.connect()
      .then(async () => {
        if (cancelled) return;
        const persisted = readPersistedTabs();
        const restored: CopilotTab[] = [];
        for (const t of persisted) {
          try {
            await gw.request("session.resume", { session_id: t.id });
            const hist = await gw.request<{ messages?: unknown }>(
              "session.history",
              { session_id: t.id },
            );
            streamingRef.current[t.id] = null;
            restored.push({
              id: t.id,
              title: t.title,
              messages: historyToMessages(hist?.messages),
              busy: false,
            });
          } catch {
            /* stale/removed session — drop it */
          }
        }
        if (cancelled) return;
        if (restored.length > 0) {
          setTabs(restored);
          setActiveId(restored[0].id);
          setBanner(null);
          return;
        }
        const created = await gw.request<{ session_id: string }>(
          "session.create",
          { close_on_disconnect: false },
        );
        if (cancelled || !created?.session_id) return;
        streamingRef.current[created.session_id] = null;
        setTabs([
          { id: created.session_id, title: "", messages: [], busy: false },
        ]);
        setActiveId(created.session_id);
        setBanner(null);
      })
      .catch((e: Error) => {
        if (!cancelled) setBanner(e.message);
      });

    return () => {
      cancelled = true;
      for (const off of offs) off();
      gw.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gw]);

  // --- actions --------------------------------------------------------------
  const send = useCallback(
    (text: string) => {
      const sid = activeIdRef.current;
      const clean = text.trim();
      if (!clean || !sid || state !== "open") return;
      const tab = tabs.find((t) => t.id === sid);
      if (tab?.busy) return;

      const assistantId = uid();
      streamingRef.current[sid] = assistantId;
      patchMessages(sid, (msgs) => [
        ...msgs,
        { id: uid(), role: "user", text: clean },
        { id: assistantId, role: "assistant", text: "", status: "thinking" },
      ]);
      setBusy(sid, true);
      gw.request("prompt.submit", { session_id: sid, text: clean }).catch(
        (e: Error) => {
          patchStreaming(sid, (m) => ({
            ...m,
            status: undefined,
            error: e.message,
          }));
          streamingRef.current[sid] = null;
          setBusy(sid, false);
        },
      );
    },
    [gw, state, tabs, patchMessages, patchStreaming, setBusy],
  );

  const stop = useCallback(() => {
    const sid = activeIdRef.current;
    if (!sid) return;
    gw.request("session.interrupt", { session_id: sid }).catch(() => undefined);
    patchStreaming(sid, (m) => ({
      ...m,
      status: undefined,
      done: true,
      text: m.text ? `${m.text}\n\n⏹ 已停止` : "⏹ 已停止",
    }));
    streamingRef.current[sid] = null;
    setBusy(sid, false);
  }, [gw, patchStreaming, setBusy]);

  const newTab = useCallback(() => {
    if (state !== "open") return;
    gw.request<{ session_id: string }>("session.create", {
      close_on_disconnect: false,
    })
      .then((created) => {
        if (!created?.session_id) return;
        streamingRef.current[created.session_id] = null;
        setTabs((prev) => [
          ...prev,
          { id: created.session_id, title: "", messages: [], busy: false },
        ]);
        setActiveId(created.session_id);
      })
      .catch(() => undefined);
  }, [gw, state]);

  const switchTab = useCallback((id: string) => setActiveId(id), []);

  const closeTab = useCallback(
    (id: string) => {
      gw.request("session.close", { session_id: id }).catch(() => undefined);
      delete streamingRef.current[id];
      setTabs((prev) => {
        const next = prev.filter((t) => t.id !== id);
        setActiveId((cur) => {
          if (cur !== id) return cur;
          const idx = prev.findIndex((t) => t.id === id);
          const fallback = next[idx] || next[idx - 1] || next[0];
          return fallback?.id ?? null;
        });
        return next;
      });
    },
    [gw],
  );

  const openHistory = useCallback(
    (session: HistorySession) => {
      // Already open → just focus it.
      const existing = tabs.find((t) => t.id === session.id);
      if (existing) {
        setActiveId(session.id);
        return;
      }
      gw.request("session.resume", { session_id: session.id })
        .then(() =>
          gw.request<{ messages?: unknown }>("session.history", {
            session_id: session.id,
          }),
        )
        .then((hist) => {
          streamingRef.current[session.id] = null;
          setTabs((prev) => [
            ...prev,
            {
              id: session.id,
              title: session.title,
              messages: historyToMessages(hist?.messages),
              busy: false,
            },
          ]);
          setActiveId(session.id);
        })
        .catch((e: Error) => setBanner(`打开历史会话失败:${e.message}`));
    },
    [gw, tabs],
  );

  const listHistory = useCallback(async () => {
    const res = await gw.request<{ sessions?: HistorySession[] }>(
      "session.list",
      { limit: 200 },
    );
    return Array.isArray(res?.sessions) ? res.sessions : [];
  }, [gw]);

  const renameSession = useCallback(
    async (id: string, title: string) => {
      await gw.request("session.title", { session_id: id, title });
      setTabs((prev) => prev.map((t) => (t.id === id ? { ...t, title } : t)));
    },
    [gw],
  );

  // Switch the active session's model in place via the gateway's `config.set`
  // (key=model). The gateway rejects this mid-turn (error 4009), so callers
  // gate on `busy`. setModel optimistically; the `session.info` event that
  // follows the switch reconfirms it.
  const switchModel = useCallback(
    async (nextModel: string, provider?: string) => {
      const id = activeIdRef.current;
      if (!id || !nextModel) return;
      // Pass the provider explicitly. Resolving a bare display name (性能/极致)
      // against the *current* provider can miss and fall back to openrouter
      // (switch_model's no-auth fallback), routing the tier to the wrong
      // gateway. `--provider` forces the correct one.
      const value = provider
        ? `${nextModel} --provider ${provider}`
        : nextModel;
      try {
        await gw.request("config.set", { key: "model", value, session_id: id });
        setModel(nextModel);
      } catch (e) {
        setBanner(e instanceof Error ? e.message : "切换模型失败");
      }
    },
    [gw],
  );

  const deleteSession = useCallback(
    async (id: string) => {
      await gw.request("session.delete", { session_id: id });
      setTabs((prev) => prev.filter((t) => t.id !== id));
    },
    [gw],
  );

  const reconnect = useCallback(() => {
    setBanner(null);
    streamingRef.current = {};
    setTabs([]);
    setActiveId(null);
    setVersion((v) => v + 1);
  }, []);

  const activeTab = useMemo(
    () => tabs.find((t) => t.id === activeId) ?? null,
    [tabs, activeId],
  );

  return {
    state,
    model,
    banner,
    tabs,
    activeId,
    activeTab,
    send,
    stop,
    newTab,
    closeTab,
    switchTab,
    switchModel,
    openHistory,
    listHistory,
    renameSession,
    deleteSession,
    reconnect,
  };
}
