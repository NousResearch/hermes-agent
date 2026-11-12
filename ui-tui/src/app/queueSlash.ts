import type { QueueItem } from '../hooks/useQueue.js'
import { t } from '../i18n/runtime.js'
import { savePendingInput } from '../lib/pendingInputs.js'

const LIST_VERBS = new Set(['list', 'ls', 'show'])
const REMOVE_VERBS = new Set(['rm', 'remove', 'delete', 'del', 'pop'])
const EDIT_VERBS = new Set(['edit', 'set'])

const USAGE_KEY = {
  add: 'slashCmd.core.queue.usageAdd',
  edit: 'slashCmd.core.queue.usageEdit',
  move: 'slashCmd.core.queue.usageMove',
  remove: 'slashCmd.core.queue.usageRemove'
} as const

export type QueuePlan =
  | { kind: 'list' }
  | { kind: 'clear' }
  | { kind: 'remove'; index: number }
  | { kind: 'edit'; index: number; text: string; display: string }
  | { kind: 'move'; from: number; to: number }
  | { kind: 'usage'; key: keyof typeof USAGE_KEY }
  | { kind: 'enqueue'; text: string; display: string }

export interface QueueSurface {
  items: QueueItem[]
  refresh: () => void
  removeAt: (index: number) => void
  rows: string[]
}

const argumentOf = (command: string) => /^\/\S+\s+([\s\S]+)$/.exec(command)?.[1] ?? ''

const splitHead = (payload: string) => {
  const head = payload.split(/\s+/, 1)[0] ?? ''

  return { head, rest: payload.slice(head.length).trim(), verb: head.toLowerCase() }
}

// Same bound as the classic CLI: a non-index token is a prompt, not a verb argument.
const indexToken = (token: string): number | null => {
  if (!/^\d+$/.test(token)) {
    return null
  }

  const value = Number(token)

  return Number.isSafeInteger(value) ? value : null
}

const enqueuePlan = (payload: string, displayPayload: string): QueuePlan => ({
  display: displayPayload || payload,
  kind: 'enqueue',
  text: payload
})

const planEdit = (rest: string, payload: string, displayPayload: string): QueuePlan => {
  if (!rest) {
    return { key: 'edit', kind: 'usage' }
  }

  const token = rest.split(/\s+/, 1)[0] ?? ''
  const index = indexToken(token)

  if (index === null) {
    return enqueuePlan(payload, displayPayload)
  }

  const text = rest.slice(token.length).trim()

  if (!text) {
    return { key: 'edit', kind: 'usage' }
  }

  // A collapsed paste keeps its label on the row and the expanded text as the prompt.
  const display = displayPayload === payload ? text : displayPayload.replace(/^\S+\s+\S+\s*/, '') || text

  return { display, index, kind: 'edit', text }
}

const planRemove = (rest: string, payload: string, displayPayload: string): QueuePlan => {
  if (!rest) {
    return { key: 'remove', kind: 'usage' }
  }

  const bits = rest.split(/\s+/)
  const index = indexToken(bits[0] ?? '')

  if (index === null) {
    return enqueuePlan(payload, displayPayload)
  }

  return bits.length === 1 ? { index, kind: 'remove' } : { key: 'remove', kind: 'usage' }
}

const planMove = (rest: string, payload: string, displayPayload: string): QueuePlan => {
  if (!rest) {
    return { key: 'move', kind: 'usage' }
  }

  const bits = rest.split(/\s+/)
  const from = indexToken(bits[0] ?? '')

  if (from === null) {
    return enqueuePlan(payload, displayPayload)
  }

  const to = bits.length === 2 ? indexToken(bits[1] ?? '') : null

  return to === null ? { key: 'move', kind: 'usage' } : { from, kind: 'move', to }
}

/** Classic CLI `/queue` routing. Anything that is not a management verb stays a prompt. */
export function planQueueSlash(displayCommand: string, expandedCommand: string): QueuePlan {
  const payload = argumentOf(expandedCommand).trim()
  const displayPayload = argumentOf(displayCommand).trim()

  if (!payload) {
    return { kind: 'list' }
  }

  const { head, rest, verb } = splitHead(payload)

  if (verb === 'add') {
    const displayRest = displayPayload.slice(head.length).trim()

    return rest ? { display: displayRest || rest, kind: 'enqueue', text: rest } : { key: 'add', kind: 'usage' }
  }

  if (LIST_VERBS.has(verb)) {
    return rest ? enqueuePlan(payload, displayPayload) : { kind: 'list' }
  }

  if (verb === 'clear') {
    return rest ? enqueuePlan(payload, displayPayload) : { kind: 'clear' }
  }

  if (EDIT_VERBS.has(verb)) {
    return planEdit(rest, payload, displayPayload)
  }

  if (REMOVE_VERBS.has(verb)) {
    return planRemove(rest, payload, displayPayload)
  }

  if (verb === 'move') {
    return planMove(rest, payload, displayPayload)
  }

  return enqueuePlan(payload, displayPayload)
}

const clip = (text: string) => {
  const flat = text.replace(/\n/g, ' ')

  return `${flat.slice(0, 80)}${flat.length > 80 ? '…' : ''}`
}

const shownRows = (surface: QueueSurface) =>
  surface.rows.length >= surface.items.length ? surface.rows : surface.items.map(item => item.display)

const listNotice = (rows: string[]) => {
  if (!rows.length) {
    return `${t('slashCmd.core.queue.empty')}  ${t('slashCmd.core.queue.usage')}`
  }

  const lines = rows.map((row, index) => `  ${index + 1}. ${clip(row)}`)

  return `${t('slashCmd.core.queue.pendingHeader', String(rows.length))}\n${lines.join('\n')}`
}

const outOfRange = (index: number, size: number) => index < 1 || index > size

const isServerRow = (index: number, surface: QueueSurface) =>
  index >= surface.items.length && index < shownRows(surface).length

const clearQueue = (surface: QueueSurface) => {
  const count = Math.max(shownRows(surface).length, surface.items.length)

  for (let index = count - 1; index >= 0; index -= 1) {
    surface.removeAt(index)
  }

  surface.refresh()

  return t(count === 1 ? 'slashCmd.core.queue.clearedOne' : 'slashCmd.core.queue.clearedOther', String(count))
}

const removeItem = (index: number, surface: QueueSurface) => {
  const rows = shownRows(surface)

  if (outOfRange(index, rows.length)) {
    return t('slashCmd.core.queue.notFound', String(index), String(rows.length))
  }

  const item = surface.items[index - 1]

  if (item?.inFlight) {
    return t('slashCmd.core.queue.alreadyStarted', String(index))
  }

  const preview = clip(item?.display ?? rows[index - 1] ?? '')

  surface.removeAt(index - 1)
  surface.refresh()

  return t('slashCmd.core.queue.removed', String(index), preview)
}

const editItem = (plan: Extract<QueuePlan, { kind: 'edit' }>, surface: QueueSurface) => {
  const size = shownRows(surface).length
  const index = plan.index - 1

  if (outOfRange(plan.index, size)) {
    return t('slashCmd.core.queue.notFound', String(plan.index), String(size))
  }

  if (isServerRow(index, surface)) {
    return t('slashCmd.core.queue.gatewayFifo')
  }

  const item = surface.items[index]

  if (!item || item.inFlight) {
    return t('slashCmd.core.queue.alreadyStarted', String(plan.index))
  }

  const previous = { display: item.display, text: item.text }

  item.text = plan.text
  item.display = plan.display

  try {
    savePendingInput(item)
  } catch (error) {
    item.text = previous.text
    item.display = previous.display

    return t('canonical.queue.notSavedDraftKept', (error as Error).message)
  }

  surface.refresh()

  return t('slashCmd.core.queue.updated', String(plan.index), clip(plan.text))
}

const restamp = (items: QueueItem[]) => {
  const base = Date.now()

  items.forEach((item, index) => {
    item.createdAt = base + index
    savePendingInput(item)
  })
}

const moveItem = (from: number, to: number, surface: QueueSurface) => {
  const size = shownRows(surface).length

  if (outOfRange(from, size) || outOfRange(to, size)) {
    return t('slashCmd.core.queue.moveOutOfRange', String(size))
  }

  if (isServerRow(from - 1, surface) || isServerRow(to - 1, surface)) {
    return t('slashCmd.core.queue.gatewayFifo')
  }

  const source = surface.items[from - 1]
  const target = surface.items[to - 1]

  if (!source || !target || source.inFlight || target.inFlight) {
    return t('slashCmd.core.queue.alreadyStarted', String(from))
  }

  if (from !== to) {
    const [moved] = surface.items.splice(from - 1, 1)

    surface.items.splice(to - 1, 0, moved!)
    restamp(surface.items)
  }

  surface.refresh()

  return t('slashCmd.core.queue.moved', String(from), String(to))
}

/** Apply a management plan to the live composer queue. Enqueue stays with the submit path. */
export function applyQueuePlan(plan: Exclude<QueuePlan, { kind: 'enqueue' }>, surface: QueueSurface): string {
  if (plan.kind === 'usage') {
    return t(USAGE_KEY[plan.key])
  }

  if (plan.kind === 'list') {
    return listNotice(shownRows(surface))
  }

  if (plan.kind === 'clear') {
    return clearQueue(surface)
  }

  if (plan.kind === 'remove') {
    return removeItem(plan.index, surface)
  }

  if (plan.kind === 'edit') {
    return editItem(plan, surface)
  }

  return moveItem(plan.from, plan.to, surface)
}
