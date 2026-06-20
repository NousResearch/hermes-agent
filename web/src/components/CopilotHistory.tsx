/**
 * CopilotHistory — the 🕐 history dropdown for the 爱马仕 Copilot.
 *
 * Lists stored conversations from the gateway (session.list), grouped by
 * recency, with search, pin, rename, delete and archive. Clicking an item
 * resumes it into a tab (handled by the parent via `onOpen`).
 *
 * pin / archive are client-side (localStorage) — the gateway has no such
 * concept; rename / delete go through the gateway session store.
 */
import { useEffect, useMemo, useRef, useState } from "react";

import type { HistorySession } from "@/hooks/useCopilotSessions";

const PINNED_KEY = "hermes.copilot.pinned.v1";
const ARCHIVED_KEY = "hermes.copilot.archived.v1";

function readIdSet(key: string): Set<string> {
  try {
    const raw = localStorage.getItem(key);
    const arr = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(arr) ? arr.filter((x) => typeof x === "string") : []);
  } catch {
    return new Set();
  }
}

function writeIdSet(key: string, set: Set<string>) {
  try {
    localStorage.setItem(key, JSON.stringify([...set]));
  } catch {
    /* best-effort */
  }
}

/** started_at may be unix seconds or ms — normalise to ms. */
function toMs(started: number): number {
  if (!started) return 0;
  return started < 1e12 ? started * 1000 : started;
}

function relativeTime(started: number): string {
  const ms = toMs(started);
  if (!ms) return "";
  const diff = Date.now() - ms;
  const min = Math.floor(diff / 60000);
  if (min < 1) return "刚刚";
  if (min < 60) return `${min} 分钟前`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} 小时前`;
  const day = Math.floor(hr / 24);
  if (day < 30) return `${day} 天前`;
  return new Date(ms).toLocaleDateString();
}

const DAY = 86_400_000;

interface Props {
  open: boolean;
  onClose: () => void;
  openTabIds: Set<string>;
  listHistory: () => Promise<HistorySession[]>;
  renameSession: (id: string, title: string) => Promise<void>;
  deleteSession: (id: string) => Promise<void>;
  onOpen: (session: HistorySession) => void;
}

const titleOf = (s: HistorySession) =>
  (s.title || s.preview || "未命名会话").trim();

export default function CopilotHistory({
  open,
  onClose,
  openTabIds,
  listHistory,
  renameSession,
  deleteSession,
  onOpen,
}: Props) {
  const [sessions, setSessions] = useState<HistorySession[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Captured at fetch time so the recency grouping stays pure across renders
  // (no Date.now() in render — react-hooks/purity).
  const [nowTs, setNowTs] = useState(0);
  const [pinned, setPinned] = useState<Set<string>>(() => readIdSet(PINNED_KEY));
  const [archived, setArchived] = useState<Set<string>>(() => readIdSet(ARCHIVED_KEY));
  const [showArchived, setShowArchived] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const panelRef = useRef<HTMLDivElement>(null);

  // (Re)load the list whenever the dropdown opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    // Showing the spinner the instant the dropdown opens is the whole point
    // here — the cascading-render guard doesn't apply to a fetch-on-open.
    /* eslint-disable react-hooks/set-state-in-effect */
    setLoading(true);
    setError(null);
    /* eslint-enable react-hooks/set-state-in-effect */
    listHistory()
      .then((rows) => {
        if (cancelled) return;
        setSessions(rows);
        setNowTs(Date.now());
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, listHistory]);

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  const togglePin = (id: string) => {
    setPinned((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      writeIdSet(PINNED_KEY, next);
      return next;
    });
  };

  const toggleArchive = (id: string) => {
    setArchived((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      writeIdSet(ARCHIVED_KEY, next);
      return next;
    });
  };

  const submitRename = async (id: string) => {
    const title = editValue.trim();
    setEditingId(null);
    if (!title) return;
    try {
      await renameSession(id, title);
      setSessions((prev) =>
        prev.map((s) => (s.id === id ? { ...s, title } : s)),
      );
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const remove = async (id: string) => {
    try {
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (pinned.has(id)) togglePin(id);
      if (archived.has(id)) toggleArchive(id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const match = (s: HistorySession) =>
      !q || titleOf(s).toLowerCase().includes(q) || s.preview.toLowerCase().includes(q);

    const visible = sessions.filter(match);
    const now = nowTs || 0;
    const pin: HistorySession[] = [];
    const recent: HistorySession[] = [];
    const older: HistorySession[] = [];
    const arch: HistorySession[] = [];

    for (const s of visible) {
      if (archived.has(s.id)) {
        arch.push(s);
        continue;
      }
      if (pinned.has(s.id)) {
        pin.push(s);
        continue;
      }
      if (now - toMs(s.started_at) <= 7 * DAY) recent.push(s);
      else older.push(s);
    }
    return { pin, recent, older, arch };
  }, [sessions, query, pinned, archived, nowTs]);

  if (!open) return null;

  const Row = (s: HistorySession) => {
    const isOpen = openTabIds.has(s.id);
    return (
      <div
        key={s.id}
        className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-[var(--copilot-control)]"
      >
        {editingId === s.id ? (
          <input
            autoFocus
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={() => submitRename(s.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submitRename(s.id);
              if (e.key === "Escape") setEditingId(null);
            }}
            className="min-w-0 flex-1 rounded border border-[var(--copilot-accent)] bg-[var(--copilot-elevated)] px-1.5 py-0.5 text-xs text-[var(--copilot-text)] outline-none"
          />
        ) : (
          <button
            type="button"
            onClick={() => {
              onOpen(s);
              onClose();
            }}
            className="flex min-w-0 flex-1 flex-col items-start text-left"
            title={titleOf(s)}
          >
            <span className="flex w-full items-center gap-1.5">
              {pinned.has(s.id) && <span className="text-[10px]">📌</span>}
              <span className="truncate text-xs font-medium text-[var(--copilot-text)]">
                {titleOf(s)}
              </span>
              {isOpen && (
                <span className="shrink-0 rounded bg-[var(--copilot-control-active)] px-1 text-[9px] text-[var(--copilot-text-tertiary)]">
                  已打开
                </span>
              )}
            </span>
            <span className="truncate text-[10px] text-[var(--copilot-text-tertiary)]">
              {relativeTime(s.started_at)}
              {s.message_count ? ` · ${s.message_count} 条` : ""}
            </span>
          </button>
        )}

        <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition group-hover:opacity-100">
          <IconBtn title={pinned.has(s.id) ? "取消置顶" : "置顶"} onClick={() => togglePin(s.id)}>
            {pinned.has(s.id) ? "📍" : "📌"}
          </IconBtn>
          <IconBtn
            title="重命名"
            onClick={() => {
              setEditingId(s.id);
              setEditValue(titleOf(s));
            }}
          >
            ✎
          </IconBtn>
          <IconBtn
            title={archived.has(s.id) ? "取消归档" : "归档"}
            onClick={() => toggleArchive(s.id)}
          >
            🗄
          </IconBtn>
          <IconBtn title="删除" onClick={() => remove(s.id)} danger>
            🗑
          </IconBtn>
        </div>
      </div>
    );
  };

  return (
    <div
      ref={panelRef}
      className="absolute right-2 top-11 z-30 flex max-h-[70vh] w-80 flex-col overflow-hidden rounded-xl border border-[var(--copilot-border)] bg-[var(--copilot-elevated)] shadow-xl"
    >
      <div className="shrink-0 border-b border-[var(--copilot-border-soft)] p-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="搜索会话…"
          className="w-full rounded-md border border-[var(--copilot-border-soft)] bg-[var(--copilot-surface)] px-2.5 py-1.5 text-xs text-[var(--copilot-text)] outline-none focus:border-[var(--copilot-accent)]"
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
        {loading && (
          <div className="px-2 py-6 text-center text-xs text-[var(--copilot-text-tertiary)]">
            加载中…
          </div>
        )}
        {error && (
          <div className="px-2 py-2 text-xs text-[var(--copilot-danger)]">⚠️ {error}</div>
        )}
        {!loading && !error && groups.pin.length === 0 && groups.recent.length === 0 && groups.older.length === 0 && groups.arch.length === 0 && (
          <div className="px-2 py-6 text-center text-xs text-[var(--copilot-text-tertiary)]">
            还没有历史会话
          </div>
        )}

        {groups.pin.length > 0 && (
          <Section label="置顶">{groups.pin.map(Row)}</Section>
        )}
        {groups.recent.length > 0 && (
          <Section label="最近 7 天">{groups.recent.map(Row)}</Section>
        )}
        {groups.older.length > 0 && (
          <Section label="更早">{groups.older.map(Row)}</Section>
        )}

        {groups.arch.length > 0 && (
          <div className="mt-1">
            <button
              type="button"
              onClick={() => setShowArchived((v) => !v)}
              className="flex w-full items-center gap-1 px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-[var(--copilot-text-tertiary)] hover:text-[var(--copilot-text)]"
            >
              <span>{showArchived ? "▾" : "▸"}</span>
              <span>已归档 ({groups.arch.length})</span>
            </button>
            {showArchived && groups.arch.map(Row)}
          </div>
        )}
      </div>
    </div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-1">
      <div className="px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-[var(--copilot-text-tertiary)]">
        {label}
      </div>
      {children}
    </div>
  );
}

function IconBtn({
  children,
  title,
  onClick,
  danger,
}: {
  children: React.ReactNode;
  title: string;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={`flex h-6 w-6 items-center justify-center rounded text-[11px] hover:bg-[var(--copilot-control-active)] ${
        danger ? "hover:text-[var(--copilot-danger)]" : ""
      }`}
    >
      {children}
    </button>
  );
}
