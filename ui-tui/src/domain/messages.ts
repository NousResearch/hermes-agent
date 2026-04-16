import { LONG_MSG } from '../config/limits.js'
import { buildToolTrailLine, fmtK } from '../lib/text.js'
import type { Msg, SessionInfo } from '../types.js'

interface ImageMeta {
  height?: number
  token_estimate?: number
  width?: number
}

interface TranscriptRow {
  context?: string
  name?: string
  role?: string
  text?: string
}

export const introMsg = (info: SessionInfo): Msg => ({ info, kind: 'intro', role: 'system', text: '' })

export const imageTokenMeta = (info: ImageMeta | null | undefined) =>
  [
    info?.width && info.height ? `${info.width}x${info.height}` : '',
    typeof info?.token_estimate === 'number' && info.token_estimate > 0 ? `~${fmtK(info.token_estimate)} tok` : ''
  ]
    .filter(Boolean)
    .join(' · ')

export const userDisplay = (text: string): string => {
  if (text.length <= LONG_MSG) {
    return text
  }

  const first = text.split('\n')[0]?.trim() ?? ''
  const words = first.split(/\s+/).filter(Boolean)
  const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80)

  return `${prefix || '(message)'} [long message]`
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

    const { context, name, role, text } = row as TranscriptRow

    if (role === 'tool') {
      pendingTools.push(buildToolTrailLine(name ?? 'tool', context ?? ''))

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
