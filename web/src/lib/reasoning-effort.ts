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
  { value: "xhigh", label: "Max" },
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

export type ReasoningMode = "standard" | "pro";

export interface ModeOption {
  value: ReasoningMode;
  label: string;
}

export const MODE_OPTIONS: ReadonlyArray<ModeOption> = [
  { value: "standard", label: "Standard" },
  { value: "pro", label: "Pro" },
];

export const VALID_MODES: ReadonlySet<string> = new Set(
  MODE_OPTIONS.map((option) => option.value),
);

/** Normalize a raw `agent.reasoning_mode` value to the provider default. */
export function normalizeMode(raw: unknown): ReasoningMode {
  const value = String(raw ?? "").trim().toLowerCase();
  return VALID_MODES.has(value) ? (value as ReasoningMode) : "standard";
}

export function isCodexProvider(provider: unknown): boolean {
  return String(provider ?? "").trim().toLowerCase() === "openai-codex";
}
