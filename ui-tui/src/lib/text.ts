import { INTERPOLATION_RE, LONG_MSG, TOOL_VERBS } from '../constants.js'

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
    .replace(/!\[(.*?)\]\(([^)\s]+)\)/g, '[image: $1]')
    .replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/_(.+?)_/g, '$1')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/==(.+?)==/g, '$1')
    .replace(/\[\^([^\]]+)\]/g, '[$1]')
    .replace(/^#{1,6}\s+/, '')
    .replace(/^\s*[-*+]\s+\[( |x|X)\]\s+/, (_m, checked: string) => `• [${checked.toLowerCase() === 'x' ? 'x' : ' '}] `)
    .replace(/^\s*[-*+]\s+/, '• ')
    .replace(/^\s*(\d+)\.\s+/, '$1. ')
    .replace(/^\s*(?:>\s*)+/, '│ ')
}

export const compactPreview = (s: string, max: number) => {
  const one = s.replace(/\s+/g, ' ').trim()

  return !one ? '' : one.length > max ? one.slice(0, max - 1) + '…' : one
}

/** Build a single tool trail line — used by both live tool.complete and resume replay. */
export const buildToolTrailLine = (name: string, context: string, error?: boolean): string => {
  const label = TOOL_VERBS[name] ?? name
  const mark = error ? '✗' : '✓'

  return `${label}${context ? ': ' + compactPreview(context, 72) : ''} ${mark}`
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
  let fence: { char: '`' | '~'; len: number } | null = null
  let rows = 0

  for (const raw of text.split('\n')) {
    const line = stripAnsi(raw)
    const maybeFence = line.match(/^\s*(`{3,}|~{3,})(.*)$/)

    if (maybeFence) {
      const marker = maybeFence[1]!
      const lang = maybeFence[2]!.trim()

      if (!fence) {
        fence = {
          char: marker[0] as '`' | '~',
          len: marker.length
        }

        if (lang) {
          rows += Math.ceil((`─ ${lang}`.length || 1) / w)
        }
      } else if (marker[0] === fence.char && marker.length >= fence.len) {
        fence = null
      }

      continue
    }

    const inCode = Boolean(fence)
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

export const isPasteBackedText = (text: string): boolean =>
  /\[\[paste:\d+\]\]|\[paste #\d+ (?:attached|excerpt)\]/.test(text)
