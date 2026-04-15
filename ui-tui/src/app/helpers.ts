import { buildToolTrailLine, fmtK, userDisplay } from '../lib/text.js'
import type { DetailsMode, Msg, SessionInfo } from '../types.js'

const DETAILS_MODES: DetailsMode[] = ['hidden', 'collapsed', 'expanded']

export interface PasteSnippet {
  label: string
  text: string
}

export const parseDetailsMode = (v: unknown): DetailsMode | null => {
  const s = typeof v === 'string' ? v.trim().toLowerCase() : ''

  return DETAILS_MODES.includes(s as DetailsMode) ? (s as DetailsMode) : null
}

export const resolveDetailsMode = (d: any): DetailsMode =>
  parseDetailsMode(d?.details_mode) ??
  { full: 'expanded' as const, collapsed: 'collapsed' as const, truncated: 'collapsed' as const }[
    String(d?.thinking_mode ?? '')
      .trim()
      .toLowerCase()
  ] ??
  'collapsed'

export const nextDetailsMode = (m: DetailsMode): DetailsMode =>
  DETAILS_MODES[(DETAILS_MODES.indexOf(m) + 1) % DETAILS_MODES.length]!

export const introMsg = (info: SessionInfo): Msg => ({ role: 'system', text: '', kind: 'intro', info })

export const shortCwd = (cwd: string, max = 28) => {
  const p = process.env.HOME && cwd.startsWith(process.env.HOME) ? `~${cwd.slice(process.env.HOME.length)}` : cwd

  return p.length <= max ? p : `…${p.slice(-(max - 1))}`
}

export const imageTokenMeta = (
  info: { height?: number; token_estimate?: number; width?: number } | null | undefined
) => {
  const dims = info?.width && info?.height ? `${info.width}x${info.height}` : ''

  const tok =
    typeof info?.token_estimate === 'number' && info.token_estimate > 0 ? `~${fmtK(info.token_estimate)} tok` : ''

  return [dims, tok].filter(Boolean).join(' · ')
}

export const looksLikeSlashCommand = (text: string) => {
  if (!text.startsWith('/')) {
    return false
  }

  const first = text.split(/\s+/, 1)[0] || ''

  return !first.slice(1).includes('/')
}

export const toTranscriptMessages = (rows: unknown): Msg[] => {
  if (!Array.isArray(rows)) {
    return []
  }

  const result: Msg[] = []
  let pendingTools: string[] = []

  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const role = (row as any).role
    const text = (row as any).text

    if (role === 'tool') {
      const name = (row as any).name ?? 'tool'
      const ctx = (row as any).context ?? ''
      pendingTools.push(buildToolTrailLine(name, ctx))

      continue
    }

    if (typeof text !== 'string' || !text.trim()) {
      continue
    }

    if (role === 'assistant') {
      const msg: Msg = { role, text }

      if (pendingTools.length) {
        msg.tools = pendingTools
        pendingTools = []
      }

      result.push(msg)

      continue
    }

    if (role === 'user' || role === 'system') {
      pendingTools = []
      result.push({ role, text })
    }
  }

  return result
}

export function fmtDuration(ms: number) {
  const total = Math.max(0, Math.floor(ms / 1000))
  const hours = Math.floor(total / 3600)
  const mins = Math.floor((total % 3600) / 60)
  const secs = total % 60

  if (hours > 0) {
    return `${hours}h ${mins}m`
  }

  if (mins > 0) {
    return `${mins}m ${secs}s`
  }

  return `${secs}s`
}

export const stickyPromptFromViewport = (
  messages: readonly Msg[],
  offsets: ArrayLike<number>,
  top: number,
  sticky: boolean
) => {
  if (sticky || !messages.length) {
    return ''
  }

  let lo = 0
  let hi = offsets.length

  while (lo < hi) {
    const mid = (lo + hi) >> 1

    if (offsets[mid]! <= top) {
      lo = mid + 1
    } else {
      hi = mid
    }
  }

  const first = Math.max(0, Math.min(messages.length - 1, lo - 1))

  if (messages[first]?.role === 'user' && (offsets[first] ?? 0) + 1 >= top) {
    return ''
  }

  for (let i = first - 1; i >= 0; i--) {
    if (messages[i]?.role !== 'user') {
      continue
    }

    if ((offsets[i] ?? 0) + 1 >= top) {
      continue
    }

    return userDisplay(messages[i]!.text.trim()).replace(/\s+/g, ' ').trim()
  }

  return ''
}
