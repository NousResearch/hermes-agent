/**
 * Pure helpers for the Projects feature.
 *
 * Deliberately framework-free so they can be exercised in the node test
 * environment without jsdom or React.
 */
import type { ProjectSession, SessionInfo } from "@/lib/api";

/**
 * Map a `ProjectSession` (from `/api/projects/{id}/sessions`) to the
 * `SessionInfo` shape that `SessionsPage` renders.
 *
 * Fields not available from the projects endpoint (message_count,
 * tool_call_count, input/output tokens, preview) are zeroed / nulled.
 * The session list still renders correctly — it just won't show token /
 * message stats for project-filtered results.
 */
export function mapProjectSessionToSessionInfo(ps: ProjectSession): SessionInfo {
  const lastActiveMs = ps.lastActive ? new Date(ps.lastActive).getTime() : 0;
  return {
    id: ps.id,
    title: ps.title || null,
    model: ps.model,
    source: null,
    started_at: lastActiveMs / 1000,
    ended_at: ps.endReason != null ? lastActiveMs / 1000 : null,
    last_active: lastActiveMs / 1000,
    is_active: ps.endReason == null,
    message_count: 0,
    tool_call_count: 0,
    input_tokens: 0,
    output_tokens: 0,
    preview: null,
  };
}

/**
 * Build the canonical path for a project detail page.
 * Both `/projects/:id` (if the route exists) and `/projects?id=<id>` are
 * supported in the router; this helper always produces the query-param form
 * so internal links stay consistent with the existing route definition.
 */
export function projectQueryPath(id: string): string {
  return `/projects?id=${encodeURIComponent(id)}`;
}

/**
 * localStorage key used by the Kanban plugin to remember the selected board.
 * Must match `LS_BOARD_KEY` in the plugin's bundled dist/index.js.
 */
const KANBAN_BOARD_LS_KEY = "hermes.kanban.selectedBoard";

/**
 * localStorage key used to pre-select a specific task when the Kanban plugin
 * mounts.  The plugin's current dist reads `hermes.kanban.selectedBoard` at
 * init time; this sibling key follows the same pattern so the plugin can
 * open the task drawer automatically once it adds support for it.
 *
 * @internal  Not yet consumed by the plugin dist — wired here so the web
 *            side is ready when the plugin is updated.
 */
export const KANBAN_TASK_LS_KEY = "hermes.kanban.selectedTask";

/**
 * Persist `boardSlug` to localStorage so the Kanban plugin selects it when
 * the /kanban page mounts.  No-op when `boardSlug` is null/undefined.
 * Swallows quota errors — the plugin handles a missing key gracefully.
 */
export function writeKanbanBoard(boardSlug: string | null | undefined): void {
  if (!boardSlug) return;
  try {
    localStorage.setItem(KANBAN_BOARD_LS_KEY, boardSlug);
  } catch {
    // quota exceeded — ignore
  }
}

/**
 * Persist `taskId` to localStorage so the Kanban plugin can open the task
 * drawer when the /kanban page mounts.  Follows the same pattern as
 * `writeKanbanBoard`.
 *
 * No-op when `taskId` is null/undefined — a missing key is treated as
 * "no pre-selected task" by the plugin.
 * Swallows quota errors — the plugin handles a missing key gracefully.
 */
export function writeKanbanTask(taskId: string | null | undefined): void {
  if (!taskId) return;
  try {
    localStorage.setItem(KANBAN_TASK_LS_KEY, taskId);
  } catch {
    // quota exceeded — ignore
  }
}

/**
 * Build the navigation path for the Kanban plugin page.
 *
 * With no arguments returns `/kanban`.
 * Passing `board` and/or `task` appends them as query params so the URL
 * carries the deep-link contract even before the plugin consumes it:
 *   kanbanPath({ board: "default" })           → /kanban?board=default
 *   kanbanPath({ board: "default", task: "t_1" }) → /kanban?board=default&task=t_1
 *
 * The plugin currently reads the board from localStorage (`writeKanbanBoard`)
 * rather than from the URL; the query params are added now so the contract is
 * in place and the plugin can pick them up without another web-side change.
 */
export function kanbanPath(options?: {
  board?: string | null;
  task?: string | null;
}): string {
  if (!options) return "/kanban";
  const params = new URLSearchParams();
  if (options.board) params.set("board", options.board);
  if (options.task) params.set("task", options.task);
  const qs = params.toString();
  return qs ? `/kanban?${qs}` : "/kanban";
}
