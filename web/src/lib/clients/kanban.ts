import { serviceFetchJSON } from "@/lib/clients/serviceFetch";
import type { DashboardServiceClientConfig } from "@/lib/clients/serviceFetch";

export const KANBAN_API_BASE_URL_ENV = "VITE_KANBAN_API_BASE_URL";

export interface KanbanTask {
  id: string;
  title: string | null;
  status: string;
  assignee: string | null;
  priority: number;
}

export interface KanbanColumn {
  status: string;
  tasks: KanbanTask[];
}

export interface KanbanStateSnapshot {
  columns: KanbanColumn[];
  tenants: string[];
  assignees: string[];
  latest_event_id: number;
  now: number;
  board?: string;
}

export interface KanbanStateOptions {
  board?: string;
}

interface RawBoardColumn {
  name: string;
  tasks: KanbanTask[];
}

interface RawBoardResponse {
  columns: RawBoardColumn[];
  tenants: string[];
  assignees: string[];
  latest_event_id: number;
  now: number;
  board?: string;
}

function normalise(raw: RawBoardResponse): KanbanStateSnapshot {
  return {
    ...raw,
    columns: raw.columns.map((column) => ({
      status: column.name,
      tasks: column.tasks,
    })),
  };
}

function buildBoardPath(board?: string): string {
  const query = board ? `?board=${encodeURIComponent(board)}` : "";
  return `/api/plugins/kanban/board${query}`;
}

export function buildKanbanEventsPath(options: KanbanStateOptions & { since: number }): string {
  const params = new URLSearchParams();
  params.set("since", String(options.since));
  if (options.board) params.set("board", options.board);
  return `/api/plugins/kanban/events?${params.toString()}`;
}

export async function fetchKanbanState(
  options: KanbanStateOptions = {},
  config?: DashboardServiceClientConfig,
): Promise<KanbanStateSnapshot> {
  const raw = await serviceFetchJSON<RawBoardResponse>(
    KANBAN_API_BASE_URL_ENV,
    buildBoardPath(options.board),
    undefined,
    config,
  );
  return normalise(raw);
}
