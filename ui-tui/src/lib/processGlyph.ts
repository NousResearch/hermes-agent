import type { ProcessRow } from '../app/processRoster.js'
import type { Theme } from '../theme.js'

// Status→glyph lookup for background-process rows; the docked panel and the
// /agents overlay render identical glyphs so the two never drift apart.

export const PROCESS_GLYPH: Record<ProcessRow['status'], { color: (t: Theme) => string; glyph: string }> = {
  running: { color: t => t.color.accent, glyph: '⚙' },
  done: { color: t => t.color.statusGood, glyph: '✔' },
  failed: { color: t => t.color.error, glyph: '✘' },
  killed: { color: t => t.color.warn, glyph: '✘' },
  lost: { color: t => t.color.muted, glyph: '?' }
}

export const processGlyph = (status: ProcessRow['status'], t: Theme): { color: string; glyph: string } => {
  const g = PROCESS_GLYPH[status]

  return { color: g.color(t), glyph: g.glyph }
}
