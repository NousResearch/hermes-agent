import type { FC } from "react";
import type { MissionSnapshot } from "./types";

interface TopStripProps {
  snapshot: MissionSnapshot;
}

function formatUsd(n: number): string {
  return n < 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(0)}`;
}

export const TopStrip: FC<TopStripProps> = ({ snapshot }) => {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 16,
        padding: "0 8px",
        height: 40,
        background: "rgba(0,0,0,0.3)",
        borderRadius: "var(--radius, 0.25rem)",
        border: "1px solid rgba(168,192,214,0.12)",
        fontFamily: "ui-monospace, monospace",
        fontSize: 12,
        flexShrink: 0,
        overflowX: "auto",
        whiteSpace: "nowrap",
      }}
    >
      {/* Model badge */}
      <span
        style={{
          background: "rgba(0,188,212,0.15)",
          border: "1px solid rgba(0,188,212,0.4)",
          borderRadius: 3,
          padding: "1px 8px",
          color: "#00bcd4",
          fontWeight: 600,
          fontSize: 11,
        }}
      >
        {snapshot.model}
      </span>

      <span style={{ color: "rgba(168,192,214,0.4)" }}>|</span>

      {/* Spend today */}
      <span style={{ color: "rgba(168,192,214,0.7)", fontSize: 11 }}>
        Today:{" "}
        <span style={{ color: "#a8c0d6", fontWeight: 600 }}>
          {formatUsd(snapshot.spendToday)}
        </span>
        <span
          title="Estimated API-equivalent spend. You're on a Max plan — actual cash spend is your flat Max subscription. This shows what these calls would have cost at API list price."
          style={{
            color: "rgba(245,166,35,0.85)",
            fontSize: 9,
            marginLeft: 4,
            padding: "0 4px",
            border: "1px solid rgba(245,166,35,0.35)",
            borderRadius: 3,
            background: "rgba(245,166,35,0.06)",
            cursor: "help",
            fontWeight: 600,
            letterSpacing: "0.04em",
          }}
        >
          est
        </span>
      </span>

      {/* Spend week */}
      <span style={{ color: "rgba(168,192,214,0.7)", fontSize: 11 }}>
        Week:{" "}
        <span style={{ color: "#a8c0d6", fontWeight: 600 }}>
          {formatUsd(snapshot.spendWeek)}
        </span>
        <span
          title="Estimated API-equivalent spend. You're on a Max plan — actual cash spend is your flat Max subscription. This shows what these calls would have cost at API list price."
          style={{
            color: "rgba(245,166,35,0.85)",
            fontSize: 9,
            marginLeft: 4,
            padding: "0 4px",
            border: "1px solid rgba(245,166,35,0.35)",
            borderRadius: 3,
            background: "rgba(245,166,35,0.06)",
            cursor: "help",
            fontWeight: 600,
            letterSpacing: "0.04em",
          }}
        >
          est
        </span>
      </span>

      <span style={{ color: "rgba(168,192,214,0.4)" }}>|</span>

      {/* Streak */}
      <span style={{ color: "rgba(168,192,214,0.7)", fontSize: 11 }}>
        Streak:{" "}
        <span
          style={{
            color: snapshot.streakDays >= 7 ? "#f5a623" : "#a8c0d6",
            fontWeight: 600,
          }}
        >
          {snapshot.streakDays}d
        </span>
        {snapshot.streakDays >= 7 && (
          <span style={{ marginLeft: 4 }} title="7+ day streak">
            🔥
          </span>
        )}
      </span>
    </div>
  );
};
