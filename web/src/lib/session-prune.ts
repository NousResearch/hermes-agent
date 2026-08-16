interface SessionPruneResult {
  removed: number;
  skipped_open: number;
}

export function formatSessionPruneResult(result: SessionPruneResult): string {
  const removed = `Pruned ${result.removed} session${result.removed === 1 ? "" : "s"}`;
  if (!result.skipped_open) return removed;

  return `${removed}. Skipped ${result.skipped_open} open session${result.skipped_open === 1 ? "" : "s"}; prune only removes ended sessions.`;
}