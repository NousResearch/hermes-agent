/**
 * Pure reasoning-effort helpers shared by the dashboard ReasoningPicker.
 *
 * Kept DOM-free so the node-environment vitest harness can cover the
 * resolution logic without loading React or the UI kit.
 *
 * Values mirror hermes_constants.VALID_REASONING_EFFORTS plus `none`
 * (thinking-off). An empty/unset config value means the Hermes default,
 * which is `medium`.
 */

export interface EffortOption {
  value: string;
  labelKey: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "ultra";
}

export const EFFORT_OPTIONS: ReadonlyArray<EffortOption> = [
  { value: "none", labelKey: "none" },
  { value: "minimal", labelKey: "minimal" },
  { value: "low", labelKey: "low" },
  { value: "medium", labelKey: "medium" },
  { value: "high", labelKey: "high" },
  { value: "xhigh", labelKey: "xhigh" },
  { value: "max", labelKey: "max" },
  { value: "ultra", labelKey: "ultra" },
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
