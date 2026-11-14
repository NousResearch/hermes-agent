import type { Theme } from '../theme.js'
import type { SubagentProgress } from '../types.js'

// Shared status→glyph lookup for the subagent surfaces. Extracted so the
// docked agents panel and the full /agents overlay render identical glyphs
// and colours — a single source of truth prevents visual drift between them.

export type SubagentStatus = SubagentProgress['status']

/** Background async delegations carry their own lifecycle vocabulary
 * (`dispatched → running → finalizing → completed|error`, plus `rejected`
 * when the capacity gate refuses one). They render through the same table so
 * a background row never falls through to the unknown-status glyph. */
export type AgentStatus = 'cancelled' | 'dispatched' | 'finalizing' | 'rejected' | SubagentStatus

export const STATUS_GLYPH: Record<AgentStatus, { color: (t: Theme) => string; glyph: string }> = {
  running: { color: t => t.color.accent, glyph: '●' },
  queued: { color: t => t.color.muted, glyph: '○' },
  dispatched: { color: t => t.color.muted, glyph: '○' },
  finalizing: { color: t => t.color.accent, glyph: '◐' },
  completed: { color: t => t.color.statusGood, glyph: '✓' },
  interrupted: { color: t => t.color.warn, glyph: '■' },
  cancelled: { color: t => t.color.warn, glyph: '■' },
  rejected: { color: t => t.color.warn, glyph: '⊘' },
  failed: { color: t => t.color.error, glyph: '✗' },
  timeout: { color: t => t.color.warn, glyph: '⌛' },
  error: { color: t => t.color.error, glyph: '⚠' }
}

/** Neutral fallback for a status this build has never heard of (an older or
 * newer daemon on the other end of the socket). Deliberately not the `error`
 * glyph: an unknown status is not a failure, and painting it red made healthy
 * rows look broken. */
const UNKNOWN_GLYPH = { color: (t: Theme) => t.color.muted, glyph: '·' }

/** Resolve a status to its glyph + theme colour, with a defensive fallback for
 * cross-version snapshots carrying an unknown status. */
export const statusGlyph = (status: string, t: Theme): { color: string; glyph: string } => {
  const g = STATUS_GLYPH[status as AgentStatus] ?? UNKNOWN_GLYPH

  return { color: g.color(t), glyph: g.glyph }
}
