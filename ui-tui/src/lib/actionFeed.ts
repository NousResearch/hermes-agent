import { compactPreview } from './text.js'

export type ActionStatus = 'error' | 'running' | 'success'

export const ACTION_FEED_VISIBLE_LIMIT = 3

export interface ParsedActionCall {
  action: string
  subject: string
  title: string
}

export interface FoldedActionDetail {
  hiddenLines: number
  preview: string
}

const CALL_RE = /^(.*?)(?:\("([\s\S]*)"\))?$/
const DURATION_RE = / \(\d+(?:\.\d)?s\)$/

const cleanSubject = (value = '') =>
  value
    .replace(/^['"]|['"]$/g, '')
    .replace(/\\n/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

const tailPath = (value: string) => {
  const cleaned = cleanSubject(value)

  if (!cleaned) {
    return ''
  }

  const parts = cleaned.split('/').filter(Boolean)
  return parts.length > 2 ? `…/${parts.slice(-2).join('/')}` : cleaned
}

const quoted = (value: string) => (value ? `"${compactPreview(cleanSubject(value), 72)}"` : '')

const ACTIONS: Record<string, (subject: string) => ParsedActionCall> = {
  'Browser Back': () => ({ action: 'Went back', subject: '', title: 'Went back' }),
  'Browser Click': subject => ({ action: 'Clicked', subject: quoted(subject), title: ['Clicked', quoted(subject)].filter(Boolean).join(' ') }),
  'Browser Console': subject => ({ action: 'Inspected console', subject: quoted(subject), title: ['Inspected console', quoted(subject)].filter(Boolean).join(' ') }),
  'Browser Navigate': subject => ({ action: 'Opened', subject: quoted(subject), title: ['Opened', quoted(subject)].filter(Boolean).join(' ') }),
  'Browser Press': subject => ({ action: 'Pressed key', subject: quoted(subject), title: ['Pressed key', quoted(subject)].filter(Boolean).join(' ') }),
  'Browser Scroll': subject => ({ action: 'Scrolled', subject: quoted(subject), title: ['Scrolled', quoted(subject)].filter(Boolean).join(' ') }),
  'Browser Snapshot': () => ({ action: 'Read page snapshot', subject: '', title: 'Read page snapshot' }),
  'Browser Type': subject => ({ action: 'Typed', subject: quoted(subject), title: ['Typed', quoted(subject)].filter(Boolean).join(' ') }),
  'Delegate Task': subject => ({ action: 'Delegated', subject: quoted(subject), title: ['Delegated', quoted(subject)].filter(Boolean).join(' ') }),
  'Execute Code': subject => ({ action: 'Ran Python', subject: quoted(subject), title: ['Ran Python', quoted(subject)].filter(Boolean).join(' ') }),
  Patch: subject => ({ action: 'Edited', subject: tailPath(subject), title: ['Edited', tailPath(subject)].filter(Boolean).join(' ') }),
  'Read File': subject => ({ action: 'Read', subject: tailPath(subject), title: ['Read', tailPath(subject)].filter(Boolean).join(' ') }),
  'Search Files': subject => ({ action: 'Searched', subject: quoted(subject), title: ['Searched', quoted(subject)].filter(Boolean).join(' ') }),
  Terminal: subject => ({ action: 'Ran', subject: quoted(subject), title: ['Ran', quoted(subject)].filter(Boolean).join(' ') }),
  Todo: () => ({ action: 'Updated todos', subject: '', title: 'Updated todos' }),
  'Write File': subject => ({ action: 'Wrote', subject: tailPath(subject), title: ['Wrote', tailPath(subject)].filter(Boolean).join(' ') })
}

export const parseActionCall = (call: string): ParsedActionCall => {
  const body = String(call || '').replace(DURATION_RE, '').trim()
  const match = body.match(CALL_RE)
  const rawName = (match?.[1] ?? body).trim()
  const subject = match?.[2] ?? ''
  const formatter = ACTIONS[rawName]

  if (formatter) {
    return formatter(subject)
  }

  const fallbackSubject = quoted(subject)
  return {
    action: rawName || 'Tool',
    subject: fallbackSubject,
    title: [rawName || 'Tool', fallbackSubject].filter(Boolean).join(' ')
  }
}

export const actionStatusGlyph = (status: ActionStatus) => {
  if (status === 'running') {
    return '●'
  }

  return status === 'error' ? '✗' : '✓'
}

export const foldActionDetail = (detail: string, maxLines = 4, maxChars = 420): FoldedActionDetail => {
  const raw = String(detail || '').trim()

  if (!raw) {
    return { hiddenLines: 0, preview: '' }
  }

  const lines = raw.split('\n')
  const visible: string[] = []
  let used = 0

  for (const line of lines) {
    const next = visible.length ? used + 1 + line.length : used + line.length

    if (visible.length >= maxLines || next > maxChars) {
      break
    }

    visible.push(line)
    used = next
  }

  if (!visible.length) {
    visible.push(compactPreview(lines[0] ?? raw, maxChars))
  }

  const hiddenLines = Math.max(0, lines.length - visible.length)
  return { hiddenLines, preview: visible.join('\n') }
}

export interface ActionFeedItemLike {
  label: string
  status: ActionStatus
}

export interface ActionFeedSelection<T extends ActionFeedItemLike> {
  hidden: number
  hiddenItems: T[]
  items: T[]
}

const IMPORTANT_ACTION_LABEL_RE = /^(?:Browser (?:Click|Navigate|Type)|Delegate Task|Execute Code|Patch|Terminal|Todo|Write File)/

export const isImportantActionLabel = (label: string): boolean => IMPORTANT_ACTION_LABEL_RE.test(label)

const actionPriority = <T extends ActionFeedItemLike>(item: T, index: number): number => {
  if (item.status === 'running') {
    return 1_000 + index
  }

  if (item.status === 'error') {
    return 900 + index
  }

  if (isImportantActionLabel(item.label)) {
    return 600 + index
  }

  // Successful Read/Search calls are useful context, but too noisy as primary
  // feed rows. Keep them available for the hidden summary unless there is
  // nothing more meaningful to show.
  return 0
}

export const summarizeHiddenActionFeedItems = <T extends ActionFeedItemLike>(
  items: readonly T[],
  maxKinds = 3,
  maxChars = 48
): string => {
  if (!items.length) {
    return ''
  }

  const counts = new Map<string, number>()

  for (const item of items) {
    const label = parseActionCall(item.label).action || 'Tool'
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }

  const entries = [...counts.entries()].sort((a, b) => b[1] - a[1])
  const visible = entries.slice(0, maxKinds).map(([label, count]) => `${label}${count > 1 ? `×${count}` : ''}`)
  const rest = entries.slice(maxKinds).reduce((sum, [, count]) => sum + count, 0)
  const summary = [...visible, rest > 0 ? `+${rest}` : ''].filter(Boolean).join(' · ')

  return compactPreview(summary, maxChars)
}

export const selectVisibleActionFeedItems = <T extends ActionFeedItemLike>(
  items: readonly T[],
  limit = ACTION_FEED_VISIBLE_LIMIT
): ActionFeedSelection<T> => {
  const visibleLimit = Math.max(1, limit)

  if (items.length <= visibleLimit) {
    return { hidden: 0, hiddenItems: [], items: [...items] }
  }

  const ranked = items
    .map((item, index) => ({ index, priority: actionPriority(item, index) }))
    .filter(({ priority }) => priority > 0)
    .sort((a, b) => b.priority - a.priority)

  const selected = (ranked.length ? ranked.slice(0, visibleLimit).map(entry => entry.index) : [items.length - 1]).sort(
    (a, b) => a - b
  )
  const selectedSet = new Set(selected)
  const hiddenItems = items.filter((_, index) => !selectedSet.has(index))

  return {
    hidden: Math.max(0, items.length - selected.length),
    hiddenItems,
    items: selected.map(index => items[index]!)
  }
}
