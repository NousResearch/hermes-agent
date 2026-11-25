import {
  DEFAULT_REASONING_EFFORT,
  REASONING_EFFORT_VALUES,
  type ReasoningEffortValue,
} from "@hermes/shared";

/**
 * Pure reasoning-effort helpers shared by the dashboard ReasoningPicker.
 *
 * Kept DOM-free so the node-environment vitest harness can cover the
 * resolution logic without loading React or the UI kit.
 *
 * Values come from @hermes/shared (hermes_constants.VALID_REASONING_EFFORTS
 * plus `none`, thinking-off). An empty/unset config value means the Hermes
 * default.
 */

export interface EffortOption {
  value: string;
  label: string;
}

const EFFORT_LABELS: Record<ReasoningEffortValue, string> = {
  none: "Off (no thinking)",
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "Extra High",
  max: "Max",
  ultra: "Ultra",
};

/** `none` first, then the seven levels ascending — the shared value order. */
export const EFFORT_OPTIONS: ReadonlyArray<EffortOption> = REASONING_EFFORT_VALUES.map(
  (value) => ({ value, label: EFFORT_LABELS[value] }),
);

export const VALID_EFFORTS: ReadonlySet<string> = new Set(REASONING_EFFORT_VALUES);

/** Normalize a raw `agent.reasoning_effort` config value to a selectable
 *  option. Empty/unknown → `medium` (Hermes' default when unset). */
export function normalizeEffort(raw: unknown): string {
  const value = String(raw ?? "").trim().toLowerCase();
  if (!value) return DEFAULT_REASONING_EFFORT;
  return VALID_EFFORTS.has(value) ? value : DEFAULT_REASONING_EFFORT;
}
