import type { Theme } from '../theme.js'
import type { SubagentProgress } from '../types.js'

// Shared status→glyph lookup for the subagent surfaces. Extracted so the
// docked agents panel and the full /agents overlay render identical glyphs
// and colours — a single source of truth prevents visual drift between them.
//
// The GLYPH comes from the active tier's table (`t.glyphs.status`), the
// COLOUR stays here: the classic renderer is the unicode tier, so colours are
// unchanged, while an ascii-tier terminal gets `[ok]`/`[x]` instead of tofu.

export type SubagentStatus = SubagentProgress['status']

/** Background async delegations carry their own lifecycle vocabulary
 * (`dispatched → running → finalizing → completed|error`, plus `rejected`
 * when the capacity gate refuses one). They render through the same table so
 * a background row never falls through to the unknown-status glyph. */
export type AgentStatus = 'cancelled' | 'dispatched' | 'finalizing' | 'rejected' | SubagentStatus

const STATUS_TONE: Record<AgentStatus, (t: Theme) => string> = {
  running: t => t.color.accent,
  queued: t => t.color.muted,
  dispatched: t => t.color.muted,
  finalizing: t => t.color.accent,
  completed: t => t.color.statusGood,
  interrupted: t => t.color.warn,
  cancelled: t => t.color.warn,
  rejected: t => t.color.warn,
  failed: t => t.color.error,
  timeout: t => t.color.warn,
  error: t => t.color.error
}

/** Neutral fallback for a status this build has never heard of (an older or
 * newer daemon on the other end of the socket). Deliberately not the `error`
 * glyph: an unknown status is not a failure, and painting it red made healthy
 * rows look broken. */
const UNKNOWN_TONE = (t: Theme) => t.color.muted

const tableFor = (status: string): AgentStatus | null => (status in STATUS_TONE ? (status as AgentStatus) : null)

/** Resolve a status to its glyph + theme colour, with a defensive fallback for
 * cross-version snapshots carrying an unknown status. */
export const statusGlyph = (status: string, t: Theme): { color: string; glyph: string } => {
  const known = tableFor(status)

  return known
    ? { color: STATUS_TONE[known](t), glyph: t.glyphs.status[known] }
    : { color: UNKNOWN_TONE(t), glyph: t.glyphs.status.unknown }
}
