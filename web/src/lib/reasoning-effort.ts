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

/** Normalize a raw `agent.reasoning_effort` config value to a selectable
 *  option. Empty/unset → `medium` (Hermes' default); thinking-off
 *  (`false`/`disabled`) → `none`; any other unknown → `medium`. */
export function normalizeEffort(raw: unknown): string {
  const value = String(raw ?? "").trim().toLowerCase();
  if (!value) return "medium";
  // A hand-written `reasoning_effort: false`/`off`/`no` reaches config as a
  // YAML boolean (String(false) === "false"), and `disabled` is the
  // spelled-out form — both mean thinking OFF. Mirror #57330, which fixed this
  // class in the Python resolvers and desktop (use-hermes-config.ts /
  // model-settings.tsx) but did not touch web/: map them to `none`, not the
  // medium fallback that would silently re-enable thinking.
  if (value === "false" || value === "disabled") return "none";
  return VALID_EFFORTS.has(value) ? value : "medium";
}
