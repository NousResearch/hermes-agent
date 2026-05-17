import type { Msg, Role } from '../types.js'

const VISIBLE_ROLES = new Set<Role>(['assistant', 'tool', 'user'])

const roleLabel = (role: Role) => {
  switch (role) {
    case 'assistant':
      return 'Hermes'

    case 'tool':
      return 'Tool'

    case 'user':
      return 'You'

    default:
      return 'System'
  }
}

const bodyForMessage = (msg: Msg) => msg.text.trim() || (msg.tools?.length ? `(${msg.tools.length} tool calls)` : '(empty)')

const clip = (text: string, limit: number) => {
  const safeLimit = Math.max(1, limit)

  return text.length > safeLimit ? `${text.slice(0, safeLimit).trimEnd()}…` : text
}

export interface TranscriptFormatOptions {
  previewChars?: number
}

export interface TranscriptExportOptions {
  sessionId?: null | string
  title?: string
}

export interface TranscriptSearchResult {
  count: number
  text: string
}

export const visibleConversationItems = (items: Msg[]) => items.filter(msg => VISIBLE_ROLES.has(msg.role))

const formatTranscriptItems = (items: Array<{ msg: Msg; visibleIndex: number }>, options: TranscriptFormatOptions = {}) => {
  const previewChars = Math.max(1, options.previewChars ?? 400)

  return items
    .map(({ msg, visibleIndex }) => {
      const tag = `${roleLabel(msg.role)} #${visibleIndex + 1}`
      const body = clip(bodyForMessage(msg), previewChars)

      return `[${tag}]\n${body}`
    })
    .join('\n\n')
}

export const formatTranscript = (items: Msg[], options: TranscriptFormatOptions = {}) =>
  formatTranscriptItems(
    visibleConversationItems(items).map((msg, visibleIndex) => ({ msg, visibleIndex })),
    options
  )

export const searchTranscript = (items: Msg[], query: string, options: TranscriptFormatOptions = {}): TranscriptSearchResult => {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return { count: 0, text: '' }
  }

  const matches = visibleConversationItems(items)
    .map((msg, visibleIndex) => ({ msg, visibleIndex }))
    .filter(({ msg }) => bodyForMessage(msg).toLowerCase().includes(needle))

  return { count: matches.length, text: formatTranscriptItems(matches, options) }
}

export const exportTranscriptJson = (items: Msg[], options: TranscriptExportOptions = {}) =>
  `${JSON.stringify(
    {
      exported_at: new Date().toISOString(),
      messages: visibleConversationItems(items).map(msg => ({ role: msg.role, text: msg.text })),
      session_id: options.sessionId ?? null,
      title: options.title?.trim() || undefined
    },
    null,
    2
  )}\n`
