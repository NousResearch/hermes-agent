import type { FC } from "react";
import type { HealthChip } from "./types";
import { HealthChipCard } from "./HealthChip";

const RUNTIME_ORDER = [
  "codex",
  "claude-code",
  "ruflo",
  "hermes",
  "kanban",
  "cron",
];

interface LeftRailProps {
  runtimes: HealthChip[];
}

export const LeftRail: FC<LeftRailProps> = ({ runtimes }) => {
  const ordered = RUNTIME_ORDER.map(
    (name) => runtimes.find((r) => r.name === name),
  ).filter(Boolean) as HealthChip[];

  // Append any runtimes not in the canonical order
  const extra = runtimes.filter((r) => !RUNTIME_ORDER.includes(r.name));

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflowY: "auto",
        padding: "4px 0",
      }}
    >
      {[...ordered, ...extra].map((chip) => (
        <HealthChipCard key={chip.name} chip={chip} />
      ))}

      {runtimes.length === 0 && (
        <div
          style={{
            fontSize: 11,
            color: "var(--color-muted-foreground, #6a8099)",
            fontFamily: "ui-monospace, monospace",
            padding: "8px 4px",
          }}
        >
          Loading runtimes…
        </div>
      )}
    </div>
  );
};
