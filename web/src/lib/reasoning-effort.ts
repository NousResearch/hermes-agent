/**
 * Pure reasoning-effort helpers shared by the dashboard ReasoningPicker.
 *
 * Kept DOM-free so the node-environment vitest harness can cover the
 * resolution logic without loading React or the UI kit.
 *
 * Selectable values mirror hermes_constants.VALID_REASONING_EFFORTS plus
 * `none` (thinking-off). Legacy values are accepted by normalizeEffort() for
 * existing configs but are not surfaced in the picker.
 */

export interface EffortOption {
  value: string;
  label: string;
}

export const EFFORT_OPTIONS: ReadonlyArray<EffortOption> = [
  { value: "none", label: "Off (no thinking)" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
  { value: "extra_high", label: "Extra High" },
];

export const VALID_EFFORTS: ReadonlySet<string> = new Set(
  EFFORT_OPTIONS.map((o) => o.value),
);

/** Normalize a raw `agent.reasoning_effort` config value to a selectable
 *  option. Empty/unknown → `medium` (Hermes' default when unset). */
export function normalizeEffort(raw: unknown): string {
  const value = String(raw ?? "").trim().toLowerCase();
  if (!value) return "medium";
  const squashed = value.replace(/[-_]+/g, " ").replace(/\s+/g, " ").trim();
  if (value === "minimal" || squashed === "minimum") return "low";
  if (["xhigh", "x high", "extra high", "max", "maximum"].includes(squashed)) {
    return "extra_high";
  }
  return VALID_EFFORTS.has(value) ? value : "medium";
}
