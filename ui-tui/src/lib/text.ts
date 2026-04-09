import { INTERPOLATION_RE, LONG_MSG } from '../constants.js'

// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;]*m/g

export const stripAnsi = (s: string) => s.replace(ANSI_RE, '')

export const hasAnsi = (s: string) => s.includes('\x1b[') || s.includes('\x1b]')

const renderEstimateLine = (line: string) => {
  const trimmed = line.trim()

  if (trimmed.startsWith('|')) {
    return trimmed
      .split('|')
      .filter(Boolean)
      .map(cell => cell.trim())
      .join('  ')
  }

  return line
    .replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/^#{1,3}\s+/, '')
    .replace(/^\s*[-*]\s+/, '• ')
    .replace(/^\s*(\d+)\.\s+/, '$1. ')
    .replace(/^>\s?/, '│ ')
}

export const compactPreview = (s: string, max: number) => {
  const one = s.replace(/\s+/g, ' ').trim()

  return !one ? '' : one.length > max ? one.slice(0, max - 1) + '…' : one
}

/** Tool completed / failed row in the inline trail (not CoT prose). */
export const isToolTrailResultLine = (line: string) => line.endsWith(' ✓') || line.endsWith(' ✗')

/** Whether a persisted/activity tool line belongs to the same tool label as a newer line. */
export const sameToolTrailGroup = (label: string, entry: string) =>
  entry === `${label} ✓` || entry === `${label} ✗` || entry.startsWith(`${label}:`)

/** Index of the last non-result trail line, or -1. */
export const lastCotTrailIndex = (trail: readonly string[]) => {
  for (let i = trail.length - 1; i >= 0; i--) {
    if (!isToolTrailResultLine(trail[i]!)) {
      return i
    }
  }

  return -1
}

export const THINKING_COT_MAX = 160
export const THINKING_COT_FADE = 5

export const thinkingCotTail = (reasoning: string) => reasoning.replace(/\n/g, ' ').slice(-THINKING_COT_MAX)

/** Scale #RRGGBB by k ∈ [0,1] — used for left-edge fade toward terminal bg. */
export const scaleHex = (hex: string, k: number) => {
  const h = hex.replace('#', '')

  const ch = (o: number) =>
    Math.round(parseInt(h.slice(o, o + 2), 16) * k)
      .toString(16)
      .padStart(2, '0')

  return `#${ch(0)}${ch(2)}${ch(4)}`
}

export const estimateRows = (text: string, w: number, compact = false) => {
  let inCode = false
  let rows = 0

  for (const raw of text.split('\n')) {
    const line = stripAnsi(raw)

    if (line.startsWith('```')) {
      if (!inCode) {
        const lang = line.slice(3).trim()

        if (lang) {
          rows += Math.ceil((`─ ${lang}`.length || 1) / w)
        }
      }

      inCode = !inCode

      continue
    }

    const trimmed = line.trim()

    if (!inCode && trimmed.startsWith('|') && /^[|\s:-]+$/.test(trimmed)) {
      continue
    }

    const rendered = inCode ? line : renderEstimateLine(line)

    if (compact && !rendered.trim()) {
      continue
    }

    rows += Math.ceil((rendered.length || 1) / w)
  }

  return Math.max(1, rows)
}

export const flat = (r: Record<string, string[]>) => Object.values(r).flat()

const COMPACT_NUMBER = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
  notation: 'compact'
})

export const fmtK = (n: number) => COMPACT_NUMBER.format(n)

export const hasInterpolation = (s: string) => {
  INTERPOLATION_RE.lastIndex = 0

  return INTERPOLATION_RE.test(s)
}

export const pick = <T>(a: T[]) => a[Math.floor(Math.random() * a.length)]!

export const userDisplay = (text: string): string => {
  if (text.length <= LONG_MSG) {
    return text
  }

  const first = text.split('\n')[0]?.trim() ?? ''
  const words = first.split(/\s+/).filter(Boolean)
  const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80)

  return `${prefix || '(message)'} [long message]`
}
