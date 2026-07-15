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

/** Restrict the generic Hermes effort menu to a model's declared levels.
 *  `max` is accepted as an API alias for Hermes' `xhigh` UI value. */
export function optionsForSupportedEfforts(
  supported?: ReadonlyArray<string>,
): ReadonlyArray<EffortOption> {
  if (!supported?.length) return EFFORT_OPTIONS;
  const allowed = new Set(
    supported.map((raw) => {
      const value = String(raw ?? "").trim().toLowerCase();
      return value === "max" ? "xhigh" : value;
    }),
  );
  const filtered = EFFORT_OPTIONS.filter((option) => allowed.has(option.value));
  return filtered.length ? filtered : EFFORT_OPTIONS;
}

/** Normalize a raw `agent.reasoning_effort` config value to a selectable
 *  option. Empty/unknown → `medium` (Hermes' default when unset). */
export function normalizeEffort(raw: unknown): string {
  const value = String(raw ?? "").trim().toLowerCase();
  if (!value) return "medium";
  return VALID_EFFORTS.has(value) ? value : "medium";
}
