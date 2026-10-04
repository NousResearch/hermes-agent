import type { MessageUserPayload } from '@hermes/shared/gateway-events'
import { LONG_MSG } from '../config/limits.js'
import { t } from '../i18n/runtime.js'
import { buildToolTrailLine } from '../lib/text.js'
import type { Msg, SessionInfo } from '../types.js'

export const introMsg = (info: SessionInfo): Msg => ({ info, kind: 'intro', role: 'system', text: '' })

export const userDisplay = (text: string) => {
  if (text.length <= LONG_MSG) {
    return text
  }

  const first = text.split('\n')[0]?.trim() ?? ''
  const words = first.split(/\s+/).filter(Boolean)
  const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80)

  return t('libText.messages.longMessage', prefix || t('libText.messages.messageFallback'))
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

    const { context, display_kind, name, role, text, timestamp } = row as TranscriptRow

    const createdAt =
      typeof timestamp === 'number' && Number.isFinite(timestamp) && timestamp > 0 ? timestamp : undefined

    if (role === 'tool') {
      pending.push(buildToolTrailLine(name ?? 'tool', context ?? ''))

      continue
    }

    if (typeof text !== 'string' || !text.trim()) {
      continue
    }

    // Display-only timeline events: render as dim ◈ markers instead of
    // opaque user messages. Hidden compaction handoffs are skipped entirely.
    if (display_kind === 'hidden') {
      continue
    }

    if (display_kind === 'model_switch') {
      out.push({ kind: 'event', role: 'system', text: t('libText.messages.modelChanged') })
      pending = []

      continue
    }

    if (display_kind === 'auto_continue') {
      out.push({ kind: 'event', role: 'system', text: t('libText.messages.resumedInterruptedTurn') })
      pending = []

      continue
    }

    if (display_kind === 'personality_switch') {
      out.push({ kind: 'event', role: 'system', text: t('libText.messages.personalityChanged') })
      pending = []

      continue
    }

    if (display_kind === 'async_delegation_complete' || display_kind === 'process_complete') {
      const meta = (row as TranscriptRow).display_metadata
      const count = meta && typeof meta.task_count === 'number' ? meta.task_count : undefined

      const label =
        display_kind === 'process_complete'
          ? t('libText.messages.backgroundProcessFinished')
          : count === undefined
            ? t('libText.messages.backgroundAgentWorkFinished')
            : t(
                count === 1
                  ? 'libText.messages.backgroundAgentsFinishedOne'
                  : 'libText.messages.backgroundAgentsFinishedOther',
                count
              )

      out.push({
        kind: 'event',
        role: 'system',
        text: typeof meta?.display_text === 'string' ? meta.display_text : label
      })
      pending = []

      continue
    }

    if (role === 'assistant') {
      out.push({ role, text, ...(createdAt !== undefined && { createdAt }), ...(pending.length && { tools: pending }) })
      pending = []
    } else if (role === 'user' || role === 'system') {
      out.push({
        role,
        text,
        rowId: (row as TranscriptRow).row_id,
        messageUid: (row as TranscriptRow).message_uid,
        ...(createdAt !== undefined && { createdAt })
      })
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

interface TranscriptRow {
  row_id?: number
  message_uid?: string
  context?: string
  display_kind?: string
  display_metadata?: { task_count?: number; [key: string]: unknown }
  name?: string
  role?: string
  text?: string
  timestamp?: number
}

/** Reconcile a dispatched canonical row with hydration or this client's optimistic send. */
export function projectUserTurn(messages: Msg[], payload: MessageUserPayload): Msg[] {
  const row = payload.message
  const matches = messages.filter(
    m =>
      m.role === 'user' &&
      ((row.row_id != null && m.rowId === row.row_id) ||
        (row.message_uid != null && m.messageUid === row.message_uid) ||
        (m.clientMessageId != null && payload.client_message_ids?.includes(m.clientMessageId)))
  )
  if (
    matches.length === 1 &&
    (matches[0].rowId ?? 0) >= (row.row_id ?? 0) &&
    matches[0].messageUid === row.message_uid
  ) {
    return messages
  }
  const local = matches.length === 1 ? matches[0] : undefined
  const user: Msg = {
    ...local,
    role: 'user',
    text: local?.clientMessageId ? local.text : (row.text ?? ''),
    rowId: row.row_id ?? undefined,
    messageUid: row.message_uid ?? undefined,
    createdAt: row.timestamp ?? undefined
  }
  return [...messages.filter(m => !matches.includes(m)), user]
}
