import type { Theme } from '../theme.js'
import type { Role } from '../types.js'

// Gutter glyphs come from the active tier (`t.glyphs`): `system`/`tool` rows
// are chrome, so an ascii-tier terminal gets `.` and `*` instead of `·`/`⚡`.
// The user/assistant gutters stay brand-authored (`t.brand.prompt` /
// `t.brand.tool`) — those are user identity, not chrome.
export const ROLE: Record<Role, (t: Theme) => { body: string; glyph: string; prefix: string }> = {
  assistant: t => ({ body: t.color.text, glyph: t.brand.tool, prefix: t.color.border }),
  system: t => ({ body: '', glyph: t.glyphs.dot, prefix: t.color.muted }),
  tool: t => ({ body: t.color.muted, glyph: t.glyphs.tool, prefix: t.color.muted }),
  user: t => ({ body: t.color.label, glyph: t.brand.prompt, prefix: t.color.label })
}
