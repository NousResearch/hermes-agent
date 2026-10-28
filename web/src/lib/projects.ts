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
 * Navigate path for the Kanban plugin page.  Always `/kanban`; the board is
 * pre-selected via `writeKanbanBoard` rather than via URL params because the
 * current plugin build does not read URL search params on mount.
 */
export function kanbanPath(): string {
  return "/kanban";
}
