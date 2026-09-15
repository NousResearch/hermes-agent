import type { Usage } from '../types.js'

export const ZERO: Usage = { calls: 0, input: 0, output: 0, total: 0 }

// Shallow-compare Usage to avoid creating a new object reference when values
// haven't changed. A fresh reference on every streaming event forces every
// $uiState subscriber (including the status rule) to re-render, which showed
// up as per-delta status-bar flicker on iTerm2 (#41480). The comparator
// iterates the union of keys generically so a future Usage field (e.g.
// active_subagents, consumed by the status rule's subagent segment) can never
// be silently dropped from the comparison.
export const usageChanged = (prev: Usage, next: Usage): boolean => {
  const prevKeys = Object.keys(prev) as Array<keyof Usage>
  const nextKeys = Object.keys(next)

  if (prevKeys.length !== nextKeys.length) {
    return true
  }

  for (const key of prevKeys) {
    if (!Object.prototype.hasOwnProperty.call(next, key) || prev[key] !== next[key]) {
      return true
    }
  }

  return false
}

// Every usage producer (session.info, the mid-turn session.usage ticker,
// message.complete, manual /compress) sends a COMPLETE snapshot, not a
// partial delta -- after compression context_* is genuinely gone, not
// "unchanged". Replacing wholesale (instead of spreading `next` over `prev`)
// is what lets an omitted field actually disappear; spreading would silently
// keep the stale value forever. Required fields float to ZERO's defaults so a
// snapshot that only reports a subset of them still satisfies the Usage type
// without resurrecting the prior object's numbers.
export const replaceUsageStable = (prev: Usage, next: Usage | undefined): Usage => {
  if (!next) {
    return prev
  }

  const snapshot: Usage = { ...ZERO, ...next }

  return usageChanged(prev, snapshot) ? snapshot : prev
}
