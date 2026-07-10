import { MAX_HISTORY } from '../config/limits.js'
import type { Msg } from '../types.js'

// Cap the in-memory transcript (issue #55594, root cause #3). The renderer is
// virtualized (only ~MAX_MOUNTED items mount at once), so this bounds the
// height-cache/estimate working set, not the mounted DOM. Overridable
// per-session via config.yaml `display.max_history` — callers pass
// `getUiState().maxHistory ?? MAX_HISTORY`. The intro message is always
// preserved so the session-start separator never scrolls off.
export const capHistory = (items: Msg[], cap: number = MAX_HISTORY): Msg[] => {
  const limit = Number.isFinite(cap) ? Math.max(1, Math.round(cap)) : MAX_HISTORY

  if (items.length <= limit) {
    return items
  }

  if (items[0]?.kind === 'intro') {
    const tailCount = limit - 1

    return tailCount > 0 ? [items[0]!, ...items.slice(-tailCount)] : [items[0]!]
  }

  return items.slice(-limit)
}
