/**
 * sessionTree.ts — pure utilities for building a hierarchy from a flat
 * session list.
 *
 * Three logical buckets:
 *   roots    — main sessions (kind === "main" | undefined) with their
 *              subagent children nested under them.
 *   workers  — kanban_worker sessions (kind === "kanban_worker"), grouped
 *              separately so they can be shown in a collapsible section.
 *   orphans  — subagent sessions whose parent is NOT in the input list
 *              (parent was filtered out / deleted). Rendered as roots but
 *              marked "no parent" in the UI.
 */

import type { SessionInfo } from "@/lib/api";

export interface TreeNode {
  session: SessionInfo;
  children: TreeNode[];
}

export interface SessionTreeResult {
  roots: TreeNode[];
  workers: SessionInfo[];
  orphans: SessionInfo[];
}

/**
 * Build a session tree from a flat array.
 *
 * Works correctly when the input is already filtered (e.g. FTS search
 * results) — subagents whose parent was filtered out end up in `orphans`.
 */
export function buildSessionTree(sessions: SessionInfo[]): SessionTreeResult {
  const byId = new Map<string, SessionInfo>(sessions.map((s) => [s.id, s]));

  const workers: SessionInfo[] = [];
  const orphans: SessionInfo[] = [];
  // Map from parent id → children nodes
  const childrenMap = new Map<string, TreeNode[]>();
  const roots: TreeNode[] = [];

  // First pass: separate kanban_workers; collect everyone else
  const nonWorkers = sessions.filter((s) => {
    if (s.kind === "kanban_worker") {
      workers.push(s);
      return false;
    }
    return true;
  });

  // Build nodes for non-workers
  const nodeMap = new Map<string, TreeNode>(
    nonWorkers.map((s) => [s.id, { session: s, children: [] }]),
  );

  // Second pass: wire parent → child relationships
  for (const s of nonWorkers) {
    const pid = s.parent_session_id ?? null;
    const isMain = !s.kind || s.kind === "main";

    if (isMain) {
      // Main sessions (or kind-less) always go to roots
      roots.push(nodeMap.get(s.id)!);
      continue;
    }

    // Subagent: must have a parent in the list
    if (pid && byId.has(pid)) {
      let siblings = childrenMap.get(pid);
      if (!siblings) {
        siblings = [];
        childrenMap.set(pid, siblings);
      }
      siblings.push(nodeMap.get(s.id)!);
    } else {
      // No pid, or pid not in list → orphan
      orphans.push(s);
    }
  }

  // Third pass: attach children to their parent nodes
  for (const [pid, kids] of childrenMap) {
    const parentNode = nodeMap.get(pid);
    if (parentNode) {
      parentNode.children.push(...kids);
    }
  }

  // Sort roots: active first, then by last_active desc
  const rankRoot = (n: TreeNode) => {
    if (n.session.is_active) return 0;
    return 1;
  };
  roots.sort((a, b) => {
    const r = rankRoot(a) - rankRoot(b);
    if (r !== 0) return r;
    return (b.session.last_active ?? 0) - (a.session.last_active ?? 0);
  });

  return { roots, workers, orphans };
}
