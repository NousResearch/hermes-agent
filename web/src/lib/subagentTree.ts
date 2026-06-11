/**
 * Subagent tree builder - ported from TUI src/lib/subagentTree.ts
 * 
 * Reconstructs the spawn tree from a flat list of subagent events.
 * Groups by parentId, sorts by depth then index within each parent.
 */

import type { SubagentProgress } from "./subagentStore";

const ROOT_KEY = "__root__";

export interface SubagentNode {
  item: SubagentProgress;
  children: SubagentNode[];
  aggregate: SubagentAggregate;
}

export interface SubagentAggregate {
  activeCount: number;
  costUsd: number;
  descendantCount: number;
  filesTouched: number;
  hotness: number;
  inputTokens: number;
  maxDepthFromHere: number;
  outputTokens: number;
  totalDuration: number;
  totalTools: number;
}

/**
 * Reconstruct the subagent spawn tree from a flat event-ordered list.
 * 
 * Grouping is by parentId; a missing parentId (or one pointing at an
 * unknown subagent) is treated as a top-level spawn of the current turn.
 * Children within a parent are sorted by depth then index.
 */
export function buildSubagentTree(
  items: readonly SubagentProgress[]
): SubagentNode[] {
  if (!items.length) {
    return [];
  }

  const byParent = new Map<string, SubagentProgress[]>();
  const known = new Set<string>();

  for (const item of items) {
    known.add(item.id);
  }

  for (const item of items) {
    const parentKey =
      item.parentId && known.has(item.parentId) ? item.parentId : ROOT_KEY;
    const bucket = byParent.get(parentKey) ?? [];
    bucket.push(item);
    byParent.set(parentKey, bucket);
  }

  for (const bucket of byParent.values()) {
    bucket.sort((a, b) => (a.depth - b.depth) || (a.index - b.index));
  }

  const build = (item: SubagentProgress): SubagentNode => {
    const kids = byParent.get(item.id) ?? [];
    const children = kids.map(build);
    return {
      aggregate: aggregate(item, children),
      children,
      item,
    };
  };

  return (byParent.get(ROOT_KEY) ?? []).map(build);
}

/**
 * Roll up counts for a node's whole subtree.
 * 
 * hotness = tools per second across the subtree — a crude proxy for
 * "how much work is happening in this branch". Used to color tree rails
 * so the eye spots the expensive branch.
 */
function aggregate(
  item: SubagentProgress,
  children: readonly SubagentNode[]
): SubagentAggregate {
  let totalTools = item.toolCount;
  let totalDuration = item.durationSeconds ?? 0;
  let descendantCount = 0;
  let activeCount = isRunning(item) ? 1 : 0;
  let maxDepthFromHere = 0;
  let inputTokens = item.inputTokens ?? 0;
  let outputTokens = item.outputTokens ?? 0;
  let costUsd = item.costUsd ?? 0;
  let filesTouched =
    (item.filesRead?.length ?? 0) + (item.filesWritten?.length ?? 0);

  for (const child of children) {
    totalTools += child.aggregate.totalTools;
    totalDuration += child.aggregate.totalDuration;
    descendantCount += child.aggregate.descendantCount + 1;
    activeCount += child.aggregate.activeCount;
    maxDepthFromHere = Math.max(
      maxDepthFromHere,
      child.aggregate.maxDepthFromHere + 1
    );
    inputTokens += child.aggregate.inputTokens;
    outputTokens += child.aggregate.outputTokens;
    costUsd += child.aggregate.costUsd;
    filesTouched += child.aggregate.filesTouched;
  }

  const hotness = totalDuration > 0 ? totalTools / totalDuration : 0;

  return {
    activeCount,
    costUsd,
    descendantCount,
    filesTouched,
    hotness,
    inputTokens,
    maxDepthFromHere,
    outputTokens,
    totalDuration,
    totalTools,
  };
}

/**
 * Check if a subagent is currently active (running or queued).
 */
function isRunning(item: Pick<SubagentProgress, "status">): boolean {
  return item.status === "running" || item.status === "queued";
}

/**
 * Flat totals across the full tree — feeds the summary chip header.
 */
export function treeTotals(
  tree: readonly SubagentNode[]
): SubagentAggregate {
  let totalTools = 0;
  let totalDuration = 0;
  let descendantCount = 0;
  let activeCount = 0;
  let maxDepthFromHere = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  let costUsd = 0;
  let filesTouched = 0;

  for (const node of tree) {
    totalTools += node.aggregate.totalTools;
    totalDuration += node.aggregate.totalDuration;
    descendantCount += node.aggregate.descendantCount + 1;
    activeCount += node.aggregate.activeCount;
    maxDepthFromHere = Math.max(
      maxDepthFromHere,
      node.aggregate.maxDepthFromHere + 1
    );
    inputTokens += node.aggregate.inputTokens;
    outputTokens += node.aggregate.outputTokens;
    costUsd += node.aggregate.costUsd;
    filesTouched += node.aggregate.filesTouched;
  }

  const hotness = totalDuration > 0 ? totalTools / totalDuration : 0;

  return {
    activeCount,
    costUsd,
    descendantCount,
    filesTouched,
    hotness,
    inputTokens,
    maxDepthFromHere,
    outputTokens,
    totalDuration,
    totalTools,
  };
}

/**
 * Format totals into a compact one-line summary.
 */
export function formatSummary(totals: SubagentAggregate): string {
  const pieces = [`d${Math.max(0, totals.maxDepthFromHere)}`];
  pieces.push(
    `${totals.descendantCount} agent${totals.descendantCount === 1 ? "" : "s"}`
  );

  if (totals.totalTools > 0) {
    pieces.push(
      `${totals.totalTools} tool${totals.totalTools === 1 ? "" : "s"}`
    );
  }

  if (totals.totalDuration > 0) {
    pieces.push(fmtDuration(totals.totalDuration));
  }

  const tokens = totals.inputTokens + totals.outputTokens;
  if (tokens > 0) {
    pieces.push(`${fmtTokens(tokens)} tok`);
  }

  if (totals.costUsd > 0) {
    pieces.push(fmtCost(totals.costUsd));
  }

  if (totals.activeCount > 0) {
    pieces.push(`⚡${totals.activeCount}`);
  }

  return pieces.join(" · ");
}

/**
 * Compact dollar amount: $0.02, $1.34, $12.4 — never > 5 chars beyond the $.
 */
export function fmtCost(usd: number): string {
  if (!Number.isFinite(usd) || usd <= 0) {
    return "";
  }

  if (usd < 0.01) {
    return "<$0.01";
  }

  if (usd < 10) {
    return `$${usd.toFixed(2)}`;
  }

  return `$${usd.toFixed(1)}`;
}

/**
 * Compact token count: 12k, 1.2k, 542.
 */
export function fmtTokens(n: number): string {
  if (!Number.isFinite(n) || n <= 0) {
    return "0";
  }

  if (n < 1000) {
    return String(Math.round(n));
  }

  if (n < 10_000) {
    return `${(n / 1000).toFixed(1)}k`;
  }

  return `${Math.round(n / 1000)}k`;
}

/**
 * Ns / Nm / Nm Ss formatter for seconds.
 */
export function fmtDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.max(0, Math.round(seconds))}s`;
  }

  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);

  return s === 0 ? `${m}m` : `${m}m ${s}s`;
}
