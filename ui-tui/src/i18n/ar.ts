import type { TuiLocaleOverlay } from './en.js'

/**
 * Arabic is registered for every presentation runtime, while its Ink wording
 * remains a future locale contribution and therefore falls back to English.
 */
export const ar = {
  catalog: {},
  status: {},
  toolVerbs: {},
  trail: {}
} satisfies TuiLocaleOverlay
