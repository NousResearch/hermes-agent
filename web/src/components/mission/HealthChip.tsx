import type { FC } from "react";
import type { HealthChip } from "./types";

interface HealthChipCardProps {
  chip: HealthChip;
}

const STATUS_STYLES: Record<
  string,
  { dot: string; label: string; bg: string }
> = {
  online: {
    dot: "var(--color-success, #00e87a)",
    label: "var(--color-success, #00e87a)",
    bg: "rgba(0,232,122,0.08)",
  },
  degraded: {
    dot: "var(--color-warning, #f5a623)",
    label: "var(--color-warning, #f5a623)",
    bg: "rgba(245,166,35,0.08)",
  },
  offline: {
    dot: "var(--color-destructive, #e84040)",
    label: "var(--color-destructive, #e84040)",
    bg: "rgba(232,64,64,0.08)",
  },
  unknown: {
    dot: "var(--color-muted-foreground, #6a8099)",
    label: "var(--color-muted-foreground, #6a8099)",
    bg: "transparent",
  },
};

export const HealthChipCard: FC<HealthChipCardProps> = ({ chip }) => {
  const s = STATUS_STYLES[chip.status] ?? STATUS_STYLES.unknown;

  return (
    <div
      style={{
        background: s.bg,
        border: "1px solid",
        borderColor: s.dot,
        borderRadius: "var(--radius, 0.25rem)",
        padding: "6px 8px",
        marginBottom: 6,
        fontFamily: "ui-monospace, monospace",
        fontSize: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span
          style={{
            display: "inline-block",
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: s.dot,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            color: "var(--color-foreground-base, #a8c0d6)",
            fontWeight: 600,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {chip.label}
        </span>
      </div>

      <div
        style={{
          display: "flex",
          gap: 8,
          marginTop: 2,
          paddingLeft: 14,
          fontSize: 10,
          color: "var(--color-muted-foreground, #6a8099)",
        }}
      >
        <span style={{ color: s.label }}>{chip.status}</span>
        {chip.latencyMs != null && <span>{chip.latencyMs.toFixed(0)}ms</span>}
        {chip.detail && (
          <span
            style={{
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              maxWidth: 100,
            }}
            title={chip.detail}
          >
            {chip.detail}
          </span>
        )}
      </div>
    </div>
  );
};
