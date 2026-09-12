interface SessionPruneResult {
  removed: number
  skipped_open: number
}

interface SessionPruneCopy {
  removed: (count: number) => string
  skippedOpen: (count: number) => string
}

const englishCopy: SessionPruneCopy = {
  removed: count => `Pruned ${count} session${count === 1 ? '' : 's'}`,
  skippedOpen: count =>
    `Skipped ${count} open session${count === 1 ? '' : 's'}; prune only removes ended sessions.`
}

export function formatSessionPruneResult(result: SessionPruneResult, copy: SessionPruneCopy = englishCopy): string {
  const removed = copy.removed(result.removed)
  if (!result.skipped_open) return removed

  return `${removed}. ${copy.skippedOpen(result.skipped_open)}`
}
