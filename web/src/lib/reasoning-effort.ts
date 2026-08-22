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

export function effortLabel(value: string): string {
  return EFFORT_OPTIONS.find((option) => option.value === value)?.label ?? value;
}

export function filterEffortOptions(
  levels: readonly string[] | null | undefined,
  savedEffort?: string,
): EffortOption[] {
  if (levels === undefined || levels === null) return EFFORT_OPTIONS.slice();
  if (levels.length === 0) return [];
  const allowed = new Set(levels);
  const options = EFFORT_OPTIONS.filter((option) => allowed.has(option.value));
  const saved = savedEffort?.trim().toLowerCase();
  if (saved && !allowed.has(saved) && !options.some((option) => option.value === saved)) {
    options.push({ value: saved, label: effortLabel(saved) });
  }
  return options;
}

/** Normalize a raw `agent.reasoning_effort` config value to a selectable
 *  option. Empty/unknown → `medium` (Hermes' default when unset). */
export function normalizeEffort(raw: unknown): string {
  const value = String(raw ?? "").trim().toLowerCase();
  if (!value) return "medium";
  return VALID_EFFORTS.has(value) ? value : "medium";
}
