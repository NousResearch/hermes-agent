import { LONG_MSG } from '../config/limits.js'
import { flattenReasoning, type ReasoningFields } from '../lib/reasoning.js'
import { buildToolTrailLine, estimateTokensRough } from '../lib/text.js'
import type { Msg, SessionInfo } from '../types.js'

export const introMsg = (info: SessionInfo): Msg => ({ info, kind: 'intro', role: 'system', text: '' })

export const userDisplay = (text: string) => {
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

  const out: Msg[] = []
  let pending: string[] = []

  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const { context, display_kind, name, role, text } = row as TranscriptRow

    if (role === 'tool') {
      pending.push(buildToolTrailLine(name ?? 'tool', context ?? ''))

      continue
    }

    const thinking = role === 'assistant' ? flattenReasoning(row as TranscriptRow) : ''

    // An extended-thinking turn is persisted with its reasoning fields and no
    // visible text, and `_history_to_messages` deliberately keeps it on the
    // wire (tui_gateway/server.py — `not content_text.strip() and not
    // has_reasoning`). Dropping it here as "empty" is what made it vanish from
    // the resumed transcript while the gateway had already sent it. Only a
    // genuinely empty row — no text and no reasoning — is skipped.
    if (typeof text !== 'string' || (!text.trim() && !thinking)) {
      continue
    }

    // Display-only timeline events: render as dim ◈ markers instead of
    // opaque user messages. Hidden compaction handoffs are skipped entirely.
    if (display_kind === 'hidden') {
      continue
    }

    if (display_kind === 'model_switch') {
      out.push({ kind: 'event', role: 'system', text: 'model changed' })
      pending = []

      continue
    }

    if (display_kind === 'auto_continue') {
      out.push({ kind: 'event', role: 'system', text: 'resumed interrupted turn' })
      pending = []

      continue
    }

    if (display_kind === 'async_delegation_complete') {
      const meta = (row as TranscriptRow).display_metadata
      const count = meta && typeof meta.task_count === 'number' ? meta.task_count : undefined

      const label =
        count === undefined
          ? 'background agent work finished'
          : `${count} background agent${count === 1 ? '' : 's'} finished`

      out.push({ kind: 'event', role: 'system', text: label })
      pending = []

      continue
    }

    if (role === 'assistant') {
      // Match the live convention (turnController.flushStreamingSegment): a
      // turn with visible text is an `assistant` message, a reasoning-only one
      // is a `trail` block, so a resumed turn renders exactly like the one the
      // user just watched stream.
      out.push({
        ...(text.trim() ? { role, text } : { kind: 'trail' as const, role: 'system' as const, text: '' }),
        ...(thinking && { thinking, thinkingTokens: estimateTokensRough(thinking) }),
        ...(pending.length && { tools: pending })
      })
      pending = []
    } else if (role === 'user' || role === 'system') {
      out.push({ role, text })
      pending = []
    }
  }

  return out
}

export const fmtDuration = (ms: number) => {
  const t = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60

  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`
}

interface TranscriptRow extends ReasoningFields {
  context?: string
  display_kind?: string
  display_metadata?: { task_count?: number; [key: string]: unknown }
  name?: string
  role?: string
  text?: string
}
