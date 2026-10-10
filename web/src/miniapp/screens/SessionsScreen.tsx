import { useEffect, useState } from "react";
import type { PaginatedSessions, SessionInfo } from "@/lib/api";
import { get } from "../api";
import { useMiniApp } from "../context";

interface SearchResult {
  session_id: string;
  snippet: string;
  source: string | null;
  model: string | null;
}

function relativeAgo(epochSeconds: number): string {
  const s = Date.now() / 1000 - epochSeconds;
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export function SessionsScreen({ onOpen }: { onOpen: (id: string) => void }) {
  const { isAdmin } = useMiniApp();
  const [sessions, setSessions] = useState<SessionInfo[] | null>(null);
  const [total, setTotal] = useState(0);
  const [limit, setLimit] = useState(6);
  const [order, setOrder] = useState<"newest" | "oldest">("newest");
  const [search, setSearch] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);

  useEffect(() => {
    get<PaginatedSessions>(`/api/sessions?limit=${limit}&offset=0&order=recent`)
      .then((r) => {
        setSessions(order === "oldest" ? [...r.sessions].reverse() : r.sessions);
        setTotal(r.total);
      })
      .catch(() => setSessions([]));
  }, [limit, order]);

  useEffect(() => {
    if (!isAdmin || !search.trim()) return;
    const t = setTimeout(() => {
      get<{ results: SearchResult[] }>(`/api/sessions/search?q=${encodeURIComponent(search)}&limit=20`)
        .then((r) => setSearchResults(r.results))
        .catch(() => setSearchResults([]));
    }, 250);
    return () => clearTimeout(t);
  }, [search, isAdmin]);

  if (!sessions) return null;

  const showingSearch = isAdmin && search.trim().length > 0;

  return (
    <div style={{ padding: "16px 14px 24px", display: "flex", flexDirection: "column", gap: 10 }}>
      {isAdmin && (
        <>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search sessions"
            style={{
              width: "100%",
              boxSizing: "border-box",
              background: "var(--card)",
              border: "1px solid var(--line)",
              borderRadius: 11,
              padding: "10px 13px",
              fontFamily: "var(--mono)",
              fontSize: 13,
              color: "var(--mid)",
              outline: "none",
            }}
          />
          {!showingSearch && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ display: "flex", border: "1px solid var(--line)", borderRadius: 9, overflow: "hidden" }}>
                <button
                  onClick={() => setOrder("newest")}
                  style={{
                    padding: "5px 11px",
                    fontFamily: "var(--mono)",
                    fontSize: 10.5,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    border: "none",
                    cursor: "pointer",
                    background: order === "newest" ? "var(--card2)" : "transparent",
                    color: order === "newest" ? "var(--mid)" : "var(--t3)",
                    whiteSpace: "nowrap",
                  }}
                >
                  Newest
                </button>
                <button
                  onClick={() => setOrder("oldest")}
                  style={{
                    padding: "5px 11px",
                    fontFamily: "var(--mono)",
                    fontSize: 10.5,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    border: "none",
                    borderLeft: "1px solid var(--line)",
                    cursor: "pointer",
                    background: order === "oldest" ? "var(--card2)" : "transparent",
                    color: order === "oldest" ? "var(--mid)" : "var(--t3)",
                    whiteSpace: "nowrap",
                  }}
                >
                  Oldest
                </button>
              </div>
              <span style={{ marginLeft: "auto", fontFamily: "var(--mono)", fontSize: 11, color: "var(--t3)", whiteSpace: "nowrap" }}>
                {total} sessions
              </span>
            </div>
          )}
        </>
      )}
      {!isAdmin && (
        <div style={{ fontSize: 11, color: "var(--t3)", padding: "0 4px" }}>Your direct-message sessions</div>
      )}

      {showingSearch ? (
        (searchResults ?? []).map((r) => (
          <div
            key={r.session_id}
            onClick={() => onOpen(r.session_id)}
            style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, padding: "12px 14px", cursor: "pointer" }}
          >
            <div
              style={{
                fontSize: 12,
                color: "var(--t2)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
              }}
            >
              {r.snippet}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 8, fontFamily: "var(--mono)", fontSize: 10, color: "var(--t3)" }}>
              {r.source && (
                <span style={{ letterSpacing: "0.08em", textTransform: "uppercase", border: "1px solid var(--line)", borderRadius: 6, padding: "2px 6px" }}>
                  {r.source}
                </span>
              )}
              {r.model && <span style={{ marginLeft: "auto" }}>{r.model}</span>}
            </div>
          </div>
        ))
      ) : (
        <>
          {sessions.map((s) => (
            <div
              key={s.id}
              onClick={() => onOpen(s.id)}
              style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: 14, padding: "12px 14px", cursor: "pointer" }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: 99,
                    flexShrink: 0,
                    background: s.is_active ? "var(--success)" : "transparent",
                    border: `1px solid ${s.is_active ? "var(--success)" : "var(--line2)"}`,
                  }}
                />
                <span
                  style={{
                    fontSize: 13.5,
                    fontWeight: 600,
                    color: "var(--mid)",
                    flex: 1,
                    minWidth: 0,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {s.title || "Untitled session"}
                </span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--t3)", flexShrink: 0, whiteSpace: "nowrap" }}>
                  {relativeAgo(s.last_active)}
                </span>
              </div>
              {s.preview && (
                <div style={{ fontSize: 12, color: "var(--t2)", marginTop: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {s.preview}
                </div>
              )}
              <div style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 8, fontFamily: "var(--mono)", fontSize: 10, color: "var(--t3)" }}>
                {s.source && (
                  <span style={{ letterSpacing: "0.08em", textTransform: "uppercase", border: "1px solid var(--line)", borderRadius: 6, padding: "2px 6px", whiteSpace: "nowrap", flexShrink: 0 }}>
                    {s.source}
                  </span>
                )}
                <span style={{ whiteSpace: "nowrap" }}>{s.message_count} msgs</span>
                {s.model && <span style={{ marginLeft: "auto", whiteSpace: "nowrap" }}>{s.model}</span>}
              </div>
            </div>
          ))}
          {isAdmin && order === "newest" && total > sessions.length && (
            <button
              onClick={() => setLimit((n) => n + 6)}
              style={{
                background: "transparent",
                border: "1px dashed var(--line2)",
                borderRadius: 11,
                padding: 10,
                fontFamily: "var(--mono)",
                fontSize: 11.5,
                color: "var(--t2)",
                cursor: "pointer",
              }}
            >
              Load more · {total - sessions.length} remaining
            </button>
          )}
        </>
      )}
    </div>
  );
}
