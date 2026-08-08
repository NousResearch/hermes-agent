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

/**
 * Filter the effort options to the levels a model actually accepts.
 *
 * `levels` semantics (mirror the backend provider hook):
 *  - undefined/null → unknown → keep the full list (never hide a capable
 *    model we don't have dial knowledge for);
 *  - [] → the model has no reasoning dial → empty list (caller hides the
 *    picker);
 *  - non-empty → only those levels.
 *
 * When the current saved effort is not in the filtered set (e.g. config says
 * `ultra` but the model only accepts up to `max`), the saved value is kept
 * selectable rather than silently dropped — the fallback selection still
 * needs a representation.
 */
export function filterEffortOptions(
  levels: readonly string[] | null | undefined,
  savedEffort?: string,
): EffortOption[] {
  if (levels === undefined || levels === null) return EFFORT_OPTIONS.slice();
  const allowed = new Set(levels);
  const options = EFFORT_OPTIONS.filter((o) => allowed.has(o.value));
  const saved = savedEffort?.trim().toLowerCase();
  if (
    saved &&
    !allowed.has(saved) &&
    !options.some((o) => o.value === saved)
  ) {
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
