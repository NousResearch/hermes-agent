/**
 * Pure reasoning-effort helpers shared by the dashboard ReasoningPicker.
 *
 * Kept DOM-free so the node-environment vitest harness can cover the
 * resolution logic without loading React or the UI kit.
 *
 * Enabled values come from a cross-language contract artifact that is
 * checked against hermes_constants.VALID_REASONING_EFFORTS by Python tests.
 * `none` is the dashboard's thinking-off option. An empty/unset config value
 * means the Hermes default, which is `medium`.
 */

import enabledReasoningEfforts from "./reasoning-effort-values.json";

export interface EffortOption {
  value: string;
  label: string;
}

const EFFORT_LABELS: Readonly<Record<string, string>> = {
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "Extra High",
  max: "Max",
};

export const EFFORT_OPTIONS: ReadonlyArray<EffortOption> = [
  { value: "none", label: "Off (no thinking)" },
  ...enabledReasoningEfforts.map((value) => ({
    value,
    label: EFFORT_LABELS[value] ?? value,
  })),
];

export const VALID_EFFORTS: ReadonlySet<string> = new Set(
  EFFORT_OPTIONS.map((o) => o.value),
);

/** Normalize a raw `agent.reasoning_effort` config value to a selectable
 *  option. Empty/unknown → `medium` (Hermes' default when unset). */
export function normalizeEffort(raw: unknown): string {
  const value = String(raw ?? "").trim().toLowerCase();
  if (!value) return "medium";
  return VALID_EFFORTS.has(value) ? value : "medium";
}
