import type { SessionInfo } from '@hermes/shared/gateway-types'

import { storedClarifyRows } from '../app/clarifyTranscript.js'
import { LONG_MSG } from '../config/limits.js'
import { t } from '../i18n/runtime.js'
import { buildToolTrailLine } from '../lib/text.js'
import type { Msg } from '../types.js'

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

type TimelineEventText = (row: TranscriptRow, text: string) => string

const backgroundFinished: TimelineEventText = ({ display_kind, display_metadata: meta }) => {
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

  return typeof meta?.display_text === 'string' ? meta.display_text : label
}

/** The system line each display-only row kind renders as. */
const TIMELINE_EVENTS: Record<string, TimelineEventText> = {
  async_delegation_complete: backgroundFinished,
  auto_continue: () => t('libText.messages.resumedInterruptedTurn'),
  model_switch: () => t('libText.messages.modelChanged'),
  personality_switch: () => t('libText.messages.personalityChanged'),
  // A plugin's card (ctx.post_card): people read its own text, not the model's factual line.
  plugin_card: ({ display_metadata: meta }, text) =>
    typeof meta?.display_text === 'string' && meta.display_text.trim() ? meta.display_text : text,
  process_complete: backgroundFinished
}

/** A resumed user row's durable message row (backtrack browsing finds its prompt by it). */
const storedRowId = (role: string, rowId: unknown): Pick<Msg, 'rowId'> =>
  role === 'user' && Number.isSafeInteger(rowId) && (rowId as number) > 0 ? { rowId: rowId as number } : {}

/** A resumed tool row joins the next reply's trail; a clarify row shows what the live turn wrote after its trail
 * (the answers bubble, or the abandoned form's system line). Returns the trail lines still pending. */
const pushToolRow = (out: Msg[], pending: string[], { content, context, name }: TranscriptRow): string[] => {
  const clarify = name === 'clarify' && content ? storedClarifyRows(content) : null

  if (!clarify) {
    return [...pending, buildToolTrailLine(name ?? 'tool', context ?? '')]
  }

  out.push({ kind: 'trail', role: 'system', text: '', tools: [...pending, clarify.trail] })
  out.push({ role: clarify.abandoned ? 'system' : 'user', text: clarify.text })

  return []
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

    const { content, context, display_kind, name, role, row_id, text, timestamp } = row as TranscriptRow

    const createdAt =
      typeof timestamp === 'number' && Number.isFinite(timestamp) && timestamp > 0 ? timestamp : undefined

    if (role === 'tool') {
      pending = pushToolRow(out, pending, { content, context, name })

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

    const timelineEvent =
      display_kind && Object.hasOwn(TIMELINE_EVENTS, display_kind) ? TIMELINE_EVENTS[display_kind] : undefined

    if (timelineEvent) {
      out.push({ kind: 'event', role: 'system', text: timelineEvent(row as TranscriptRow, text) })
      pending = []

      continue
    }

    if (role === 'assistant') {
      out.push({ role, text, ...(createdAt !== undefined && { createdAt }), ...(pending.length && { tools: pending }) })
      pending = []
    } else if (role === 'user' || role === 'system') {
      out.push({ role, text, ...(createdAt !== undefined && { createdAt }), ...storedRowId(role, row_id) })
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
  content?: string
  context?: string
  display_kind?: string
  display_metadata?: { task_count?: number; [key: string]: unknown }
  name?: string
  role?: string
  row_id?: number
  text?: string
  timestamp?: number
}
