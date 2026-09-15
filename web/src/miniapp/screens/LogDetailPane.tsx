import { useEffect, useRef, useState } from "react";
import type { LogsResponse } from "@/lib/api";
import { get } from "../api";

// Debounced against GET /api/logs?search=... (hermes_cli/web_server.py's
// pre-existing get_logs, reused as-is -- not a Mini App-only reimplementation).
// Server-side search rather than "load everything once and filter client
// side": agent.log/gateway.log rotate at 5MB, so a client-side-only search
// would either transfer megabytes to a phone or silently only search the
// last couple hundred lines. Querying the server lets a search reach up to
// 2000 matching lines (get_logs's own cap) instead of whatever fit in one
// initial fetch.
const SEARCH_DEBOUNCE_MS = 300;

export function LogDetailPane({ fileKey }: { fileKey: string }) {
  const [lines, setLines] = useState<string[] | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setLoading(true);
      const params = new URLSearchParams({ file: fileKey, lines: "500" });
      if (query.trim()) params.set("search", query.trim());
      get<LogsResponse>(`/api/logs?${params.toString()}`)
        .then((resp) => setLines(resp.lines))
        .catch(() => setLines([]))
        .finally(() => setLoading(false));
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(debounceRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileKey, query]);

  return (
    <div style={{ padding: "12px 14px 24px", display: "flex", flexDirection: "column", gap: 10, height: "100%" }}>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search this log…"
        style={{
          width: "100%",
          padding: "10px 13px",
          borderRadius: 11,
          border: "1px solid var(--line2)",
          background: "var(--card)",
          color: "var(--mid)",
          fontSize: 13.5,
          fontFamily: "var(--mono)",
        }}
      />
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: 10,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--t3)",
        }}
      >
        {loading ? "Loading…" : lines ? `${lines.length} line${lines.length === 1 ? "" : "s"}` : ""}
      </div>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          background: "var(--bg)",
          border: "1px solid var(--line)",
          borderRadius: 11,
          padding: "10px 12px",
        }}
      >
        {lines && lines.length === 0 && !loading && (
          <div style={{ fontSize: 12.5, color: "var(--t3)", padding: "6px 2px" }}>
            {query.trim() ? "No matching lines." : "This log is empty."}
          </div>
        )}
        {(lines ?? []).map((line, i) => (
          <div
            key={i}
            style={{
              fontFamily: "var(--mono)",
              fontSize: 10.5,
              lineHeight: 1.6,
              color: "var(--t2)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              userSelect: "text",
            }}
          >
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}
