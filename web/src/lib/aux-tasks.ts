import type { AuxiliaryTaskAssignment } from "./api";

export interface AuxTaskMeta {
  key: string;
  label: string;
  hint: string;
  /** Plugin task that follows this slot until it is pinned itself. */
  inheritFrom?: string;
}

export const AUTO_ROUTE_EXPLANATION =
  "auto follows Default when configured, otherwise the main model.";

// Built-in auxiliary tasks, in the order `_AUX_TASK_SLOTS`
// (hermes_cli/web_server_config.py) serves them. Plugin-registered tasks are
// not listed here: the backend appends them to `/api/model/auxiliary` with the
// plugin's own `label`/`hint` and `auxTaskRows` folds them in after these.
export const BUILTIN_AUX_TASKS: readonly AuxTaskMeta[] = [
  { key: "default", label: "Default", hint: "Shared route for unpinned tasks" },
  { key: "vision", label: "Vision", hint: "Image analysis" },
  { key: "compression", label: "Compression", hint: "Context compaction" },
  { key: "approval", label: "Approval", hint: "Smart auto-approve" },
  { key: "mcp", label: "MCP", hint: "MCP tool routing" },
  { key: "title_generation", label: "Title Gen", hint: "Session titles" },
  { key: "review", label: "Review", hint: "/review subagent" },
  { key: "voice_chat", label: "Voice Chat", hint: "Spoken voice-mode replies" },
  { key: "triage_specifier", label: "Triage Specifier", hint: "Kanban spec fleshing" },
  { key: "kanban_decomposer", label: "Kanban Decomposer", hint: "Task decomposition" },
  { key: "profile_describer", label: "Profile Describer", hint: "Auto profile descriptions" },
  { key: "curator", label: "Curator", hint: "Skill-usage review" },
] as const;

/**
 * Rows the Models page renders: the built-ins, then every task the backend
 * reported that is not a built-in — plugin-registered auxiliary tasks
 * (`PluginContext.register_auxiliary_task`), labelled by the plugin. Older
 * backends never send extra rows, so this is a no-op against them; built-ins
 * stay first so the layout is stable.
 */
export function auxTaskRows(
  tasks: readonly AuxiliaryTaskAssignment[] | null | undefined,
): AuxTaskMeta[] {
  const builtin = new Set(BUILTIN_AUX_TASKS.map((t) => t.key));
  const extra = (tasks ?? [])
    .filter((entry) => !builtin.has(entry.task))
    .map((entry) => ({
      key: entry.task,
      label: entry.label || entry.task,
      hint: entry.hint || "",
      ...(entry.inherit_from ? { inheritFrom: entry.inherit_from } : {}),
    }));
  return extra.length ? [...BUILTIN_AUX_TASKS, ...extra] : [...BUILTIN_AUX_TASKS];
}

export function auxTaskLabel(
  tasks: readonly AuxiliaryTaskAssignment[] | null | undefined,
  key: string,
): string {
  return auxTaskRows(tasks).find((t) => t.key === key)?.label ?? key;
}

/** Human-readable inherited route, or null when it still resolves to main/auto. */
export function effectiveAuxRoute(
  assignment: AuxiliaryTaskAssignment | null | undefined,
): string | null {
  const route = assignment?.effective;
  if (!route) return null;
  const provider = route.provider?.trim() || "";
  const model = route.model?.trim() || "";
  const baseUrl = route.base_url?.trim() || "";
  if ((!provider || provider === "auto") && !model && !baseUrl) return null;
  return `${provider || "custom"} · ${model || "(provider default)"}`;
}

/** Runtime-compatible stored-route check for dashboard labels. */
export function auxTaskIsUnpinned(
  assignment: AuxiliaryTaskAssignment | null | undefined,
): boolean {
  if (!assignment) return true;
  const provider = assignment.provider?.trim().toLowerCase() || "";
  return (!provider || provider === "auto")
    && !assignment.model?.trim()
    && !assignment.base_url?.trim();
}

export function auxTaskRouteLabel(
  meta: AuxTaskMeta,
  assignment: AuxiliaryTaskAssignment | null | undefined,
  tasks: readonly AuxiliaryTaskAssignment[] | null | undefined,
): string {
  const isAuto = auxTaskIsUnpinned(assignment);
  const effective = effectiveAuxRoute(assignment);
  if (isAuto && assignment?.inherits_default && effective && meta.key !== "default") {
    return `inherits Default · ${effective}`;
  }
  if (isAuto && meta.inheritFrom) {
    return `inherits ${auxTaskLabel(tasks, meta.inheritFrom)} · ${effective || "auto (use main model)"}`;
  }
  if (isAuto && effective) return effective;
  if (isAuto) return "auto (use main model)";
  return `${assignment?.provider} · ${assignment?.model || "(provider default)"}`;
}
