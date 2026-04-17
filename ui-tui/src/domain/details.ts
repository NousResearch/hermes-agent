import type { DetailsMode } from '../types.js'

const MODES = ['hidden', 'collapsed', 'expanded'] as const

const THINKING_FALLBACK: Record<string, DetailsMode> = {
  collapsed: 'collapsed',
  full: 'expanded',
  truncated: 'collapsed'
}

export const parseDetailsMode = (v: unknown): DetailsMode | null => {
  const s = typeof v === 'string' ? v.trim().toLowerCase() : ''

  return MODES.find(m => m === s) ?? null
}

export const resolveDetailsMode = (d?: { details_mode?: unknown; thinking_mode?: unknown } | null): DetailsMode =>
  parseDetailsMode(d?.details_mode) ??
  THINKING_FALLBACK[
    String(d?.thinking_mode ?? '')
      .trim()
      .toLowerCase()
  ] ??
  'collapsed'

export const nextDetailsMode = (m: DetailsMode): DetailsMode => MODES[(MODES.indexOf(m) + 1) % MODES.length]!
