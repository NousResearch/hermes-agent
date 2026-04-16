import type { DetailsMode } from '../types.js'

const DETAILS_MODES: DetailsMode[] = ['hidden', 'collapsed', 'expanded']

const THINKING_FALLBACK: Record<string, DetailsMode> = {
  collapsed: 'collapsed',
  full: 'expanded',
  truncated: 'collapsed'
}

export const parseDetailsMode = (v: unknown): DetailsMode | null => {
  const s = typeof v === 'string' ? v.trim().toLowerCase() : ''

  return DETAILS_MODES.includes(s as DetailsMode) ? (s as DetailsMode) : null
}

export const resolveDetailsMode = (
  d: { details_mode?: unknown; thinking_mode?: unknown } | null | undefined
): DetailsMode =>
  parseDetailsMode(d?.details_mode) ??
  THINKING_FALLBACK[
    String(d?.thinking_mode ?? '')
      .trim()
      .toLowerCase()
  ] ??
  'collapsed'

export const nextDetailsMode = (m: DetailsMode): DetailsMode =>
  DETAILS_MODES[(DETAILS_MODES.indexOf(m) + 1) % DETAILS_MODES.length]!
