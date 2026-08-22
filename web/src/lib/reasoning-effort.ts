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
  label: string;
}

export const EFFORT_OPTIONS: ReadonlyArray<EffortOption> = [
  { value: "none", label: "Off (no thinking)" },
  { value: "minimal", label: "Minimal" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
  { value: "xhigh", label: "Extra High" },
  { value: "max", label: "Max" },
  { value: "ultra", label: "Ultra" },
];

export const VALID_EFFORTS: ReadonlySet<string> = new Set(
  EFFORT_OPTIONS.map((o) => o.value),
);

/** Filter the picker to an exact model vocabulary when one is known. */
export function effortOptionsForModel(
  efforts?: readonly string[],
  canDisableReasoning?: boolean,
): ReadonlyArray<EffortOption> {
  if (efforts === undefined && canDisableReasoning === undefined) {
    return EFFORT_OPTIONS;
  }

  const allowed = efforts === undefined
    ? new Set(EFFORT_OPTIONS.filter((option) => option.value !== "none").map((option) => option.value))
    : new Set(efforts.map((value) => value.trim().toLowerCase()));
  const enabled = EFFORT_OPTIONS.filter(
    (option) => option.value !== "none" && allowed.has(option.value),
  );
  const levels = enabled.length > 0
    ? enabled
    : [EFFORT_OPTIONS.find((option) => option.value === "medium")!];

  return canDisableReasoning === true ? [EFFORT_OPTIONS[0], ...levels] : levels;
}

/** Normalize a raw `agent.reasoning_effort` config value to a selectable
 *  option. Empty/unknown → `medium` (Hermes' default when unset). */
export function normalizeEffort(raw: unknown): string {
  const value = String(raw ?? "").trim().toLowerCase();
  if (!value) return "medium";
  return VALID_EFFORTS.has(value) ? value : "medium";
}
