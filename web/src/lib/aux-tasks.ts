import type { AuxiliaryTaskAssignment } from "./api";

export interface AuxTaskMeta {
  key: string;
  label: string;
  hint: string;
}

// Built-in auxiliary tasks, in the order `_AUX_TASK_SLOTS`
// (hermes_cli/web_server_config.py) serves them. Plugin-registered tasks are
// not listed here: the backend appends them to `/api/model/auxiliary` with the
// plugin's own `label`/`hint` and `auxTaskRows` folds them in after these.
export const BUILTIN_AUX_TASKS: readonly AuxTaskMeta[] = [
  { key: "vision", label: "Vision", hint: "Image analysis" },
  { key: "compression", label: "Compression", hint: "Context compaction" },
  { key: "skills_hub", label: "Skills Hub", hint: "Skill search" },
  { key: "approval", label: "Approval", hint: "Smart auto-approve" },
  { key: "mcp", label: "MCP", hint: "MCP tool routing" },
  { key: "title_generation", label: "Title Gen", hint: "Session titles" },
  { key: "review", label: "Review", hint: "/review subagent" },
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
    }));
  return extra.length ? [...BUILTIN_AUX_TASKS, ...extra] : [...BUILTIN_AUX_TASKS];
}

export function auxTaskLabel(
  tasks: readonly AuxiliaryTaskAssignment[] | null | undefined,
  key: string,
): string {
  return auxTaskRows(tasks).find((t) => t.key === key)?.label ?? key;
}
