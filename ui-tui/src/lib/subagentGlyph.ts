import type { Theme } from '../theme.js'
import type { SubagentProgress } from '../types.js'

// Shared status→glyph lookup for the subagent surfaces. Extracted so the
// docked agents panel and the full /agents overlay render identical glyphs
// and colours — a single source of truth prevents visual drift between them.

export type SubagentStatus = SubagentProgress['status']

export const STATUS_GLYPH: Record<SubagentStatus, { color: (t: Theme) => string; glyph: string }> = {
  running: { color: t => t.color.accent, glyph: '●' },
  queued: { color: t => t.color.muted, glyph: '○' },
  completed: { color: t => t.color.statusGood, glyph: '✓' },
  interrupted: { color: t => t.color.warn, glyph: '■' },
  failed: { color: t => t.color.error, glyph: '✗' },
  timeout: { color: t => t.color.warn, glyph: '⌛' },
  error: { color: t => t.color.error, glyph: '⚠' }
}

/** Resolve a status to its glyph + theme colour, with a defensive fallback for
 * cross-version snapshots carrying an unknown status. */
export const statusGlyph = (status: string, t: Theme): { color: string; glyph: string } => {
  const g = STATUS_GLYPH[status as SubagentStatus] ?? STATUS_GLYPH.error

  return { color: g.color(t), glyph: g.glyph }
}
