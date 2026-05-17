import type { FC } from "react";
import type { MissionSnapshot } from "./types";

interface RightRailProps {
  snapshot: MissionSnapshot;
}

function timeAgo(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diffMs / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

function timeUntil(iso: string): string {
  const diffMs = new Date(iso).getTime() - Date.now();
  if (diffMs <= 0) return "now";
  const m = Math.floor(diffMs / 60000);
  if (m < 60) return `in ${m}m`;
  return `in ${Math.floor(m / 60)}h ${m % 60}m`;
}

const SECTION_STYLE: React.CSSProperties = {
  marginBottom: 12,
  paddingBottom: 10,
  borderBottom: "1px solid rgba(168,192,214,0.1)",
};

const HEADING_STYLE: React.CSSProperties = {
  fontSize: 9,
  fontFamily: "ui-monospace, monospace",
  color: "rgba(168,192,214,0.4)",
  letterSpacing: "0.12em",
  marginBottom: 6,
  textTransform: "uppercase" as const,
};

export const RightRail: FC<RightRailProps> = ({ snapshot }) => {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflowY: "auto",
        padding: "4px 6px",
        fontFamily: "ui-monospace, monospace",
      }}
    >
      {/* Recent sessions */}
      <div style={SECTION_STYLE}>
        <div style={HEADING_STYLE}>Recent Sessions</div>
        {snapshot.recentSessions.length === 0 ? (
          <div style={{ fontSize: 11, color: "rgba(168,192,214,0.4)" }}>
            No sessions
          </div>
        ) : (
          snapshot.recentSessions.slice(0, 5).map((s) => (
            <div
              key={s.id}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                marginBottom: 6,
                gap: 6,
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  color: "var(--color-foreground-base, #a8c0d6)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  flex: 1,
                }}
                title={s.preview}
              >
                &ldquo;{s.preview.slice(0, 40)}{s.preview.length > 40 ? "…" : ""}&rdquo;
              </div>
              <div
                style={{
                  fontSize: 10,
                  color: "rgba(168,192,214,0.5)",
                  flexShrink: 0,
                }}
              >
                {timeAgo(s.createdAt)}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Last dream */}
      {snapshot.lastDream && (
        <div style={SECTION_STYLE}>
          <div style={HEADING_STYLE}>Last Dream</div>
          <div
            style={{
              fontSize: 11,
              color: "rgba(171,71,188,0.9)",
              lineHeight: 1.45,
              maxHeight: 72,
              overflow: "hidden",
            }}
          >
            &ldquo;{snapshot.lastDream.slice(0, 120)}{snapshot.lastDream.length > 120 ? "…" : ""}&rdquo;
          </div>
        </div>
      )}

      {/* Next cron */}
      {snapshot.nextCron && (
        <div style={{ marginBottom: 8 }}>
          <div style={HEADING_STYLE}>Next Cron</div>
          <div
            style={{
              fontSize: 11,
              color: "var(--color-foreground-base, #a8c0d6)",
              marginBottom: 2,
            }}
          >
            {snapshot.nextCron.name}
          </div>
          <div style={{ display: "flex", gap: 8, fontSize: 10 }}>
            <span style={{ color: "var(--color-warning, #f5a623)" }}>
              {timeUntil(snapshot.nextCron.nextRun)}
            </span>
            <span style={{ color: "rgba(168,192,214,0.4)" }}>
              {snapshot.nextCron.schedule}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
