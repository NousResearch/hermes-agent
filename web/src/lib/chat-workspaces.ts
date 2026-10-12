/**
 * Workspace picker plumbing for the dashboard Chat tab: which host directory a
 * FRESH chat starts in (`/api/pty?cwd=`), persisted per management profile so
 * a phone remembers the repo it drives.
 */

import type { ChatWorkspacesResponse } from "@/lib/api";

export interface WorkspaceOption {
  value: string;
  label: string;
}

const STORAGE_PREFIX = "hermes-chat-workspace:";

export function workspaceStorageKey(profile?: string): string {
  return `${STORAGE_PREFIX}${profile ?? ""}`;
}

export function readStoredWorkspace(profile?: string): string {
  try {
    return localStorage.getItem(workspaceStorageKey(profile)) ?? "";
  } catch {
    return "";
  }
}

export function writeStoredWorkspace(profile: string | undefined, cwd: string): void {
  try {
    const key = workspaceStorageKey(profile);
    if (cwd) localStorage.setItem(key, cwd);
    else localStorage.removeItem(key);
  } catch {
    // Storage unavailable (private mode / quota): the pick still applies this session.
  }
}

/** `~`-abbreviate a host path for compact display. */
export function abbreviateHomePath(path: string, home: string): string {
  const h = home.replace(/[\\/]+$/, "");
  if (!h) return path;
  if (path === h) return "~";
  const sep = path.startsWith(h + "/") ? "/" : path.startsWith(h + "\\") ? "\\" : null;
  return sep ? `~${sep}${path.slice(h.length + 1)}` : path;
}

/**
 * Flatten the workspaces payload into picker rows: explicit project folders
 * first (a multi-folder project contributes one row per folder), then
 * discovered repos not already covered by a project folder, most recently
 * active first. Deduped on the path.
 */
export function workspaceOptions(data: ChatWorkspacesResponse): WorkspaceOption[] {
  const seen = new Set<string>();
  const out: WorkspaceOption[] = [];
  const push = (value: string, label: string) => {
    if (!value || seen.has(value)) return;
    seen.add(value);
    out.push({ value, label });
  };
  for (const project of data.projects) {
    if (project.archived) continue;
    const folders = project.folders.length
      ? project.folders
      : project.primary_path
        ? [{ path: project.primary_path, label: null, is_primary: true }]
        : [];
    for (const folder of folders) {
      const suffix =
        folders.length > 1
          ? ` · ${folder.label || abbreviateHomePath(folder.path, data.home)}`
          : "";
      push(folder.path, `${project.name}${suffix}`);
    }
  }
  const repos = [...data.repos].sort((a, b) => b.last_active - a.last_active);
  for (const repo of repos) {
    push(repo.root, `${repo.label}  ${abbreviateHomePath(repo.root, data.home)}`);
  }
  return out;
}
