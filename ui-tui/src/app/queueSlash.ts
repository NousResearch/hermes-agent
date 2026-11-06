import type { QueueItem } from '../hooks/useQueue.js'
import { t } from '../i18n/runtime.js'
import { savePendingInput } from '../lib/pendingInputs.js'

import { isQueueItemProtected, parseQueueCommand, type QueueCommand } from './queueCommands.js'
import type { SlashRunCtx } from './slash/types.js'
import { captureDestination, isCurrentDestination, type SubmissionDestination } from './submissionDestination.js'
import { getUiState } from './uiStore.js'

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

const PREVIEW_CHARS = 50

export const previewOf = (text: string): string => {
  const flat = text.replace(/\n/g, ' ')

  return `${flat.slice(0, PREVIEW_CHARS)}${flat.length > PREVIEW_CHARS ? '…' : ''}`
}

const argumentOf = (command: string) => /^\/\S+\s+([\s\S]+)$/.exec(command)?.[1] ?? ''

const enqueuePlan = (payload: string, displayPayload: string): QueuePlan => ({
  display: displayPayload || payload,
  kind: 'enqueue',
  text: payload
})

const resolvePayloads = (displayPayload: string, payload: string, pattern: RegExp, fallback: string) => {
  const text = payload.replace(pattern, '') || fallback
  const display = displayPayload === payload ? text : displayPayload.replace(pattern, '') || text

  return { display, text }
}

const planUsage = (displayPayload: string): QueuePlan => {
  const head = displayPayload.split(/\s+/, 1)[0]?.toLowerCase() ?? ''

  if (head === 'edit' || head === 'set') {
    return { key: 'edit', kind: 'usage' }
  }

  if (head === 'move') {
    return { key: 'move', kind: 'usage' }
  }

  if (head === 'add') {
    return { key: 'add', kind: 'usage' }
  }

  return { key: 'remove', kind: 'usage' }
}

const planFromCommand = (cmd: QueueCommand, displayPayload: string, payload: string): QueuePlan => {
  switch (cmd.kind) {
    case 'prompt':
      return enqueuePlan(payload, displayPayload)

    case 'list':
      return { kind: 'list' }

    case 'clear':
      return { kind: 'clear' }

    case 'remove':
      return { index: cmd.index, kind: 'remove' }
    case 'edit': {
      const { display, text } = resolvePayloads(displayPayload, payload, /^\S+\s+\S+\s*/, cmd.text)

      return { display, index: cmd.index, kind: 'edit', text }
    }

    case 'move':
      return { from: cmd.from, kind: 'move', to: cmd.to }
    case 'add': {
      const { display, text } = resolvePayloads(displayPayload, payload, /^\S+\s*/, cmd.text)

      return { display, kind: 'enqueue', text }
    }

    case 'error':
      return planUsage(displayPayload)
  }
}

/** Classic CLI `/queue` routing. Anything that is not a management verb stays a prompt. */
export function planQueueSlash(displayCommand: string, expandedCommand: string): QueuePlan {
  const displayPayload = argumentOf(displayCommand).trim()
  const payload = argumentOf(expandedCommand).trim()

  if (!displayPayload && !payload) {
    return { kind: 'list' }
  }

  const cmd = parseQueueCommand(displayPayload)

  return planFromCommand(cmd, displayPayload, payload)
}

const isProtected = (item: QueueItem): boolean =>
  Boolean(item.inFlight || isQueueItemProtected(item, getUiState().info?.pending_submissions))

export const listQueue = (items: QueueItem[], destination?: SubmissionDestination): string => {
  if (destination && !isCurrentDestination(destination)) {
    return t('slashCmd.core.queue.changed')
  }

  if (!items.length) {
    return t('slashCmd.core.queue.empty')
  }

  const receipts = getUiState().info?.pending_submissions

  const rows = items.map((item, index) => {
    const status = item.inFlight
      ? t('slashCmd.core.queue.inFlight')
      : isQueueItemProtected(item, receipts)
        ? t('slashCmd.core.queue.authorityOwned')
        : item.failed
          ? t('slashCmd.core.queue.unconfirmed')
          : t('slashCmd.core.queue.pending')

    return `${index + 1}. [${status}] ${previewOf(item.display ?? item.text)}`
  })

  return [t('slashCmd.core.queue.localHeader'), ...rows].join('\n')
}

export const removeQueueItem = (
  index: number,
  items: QueueItem[],
  removeQueue: (i: number) => void,
  refresh?: () => void,
  destination?: SubmissionDestination,
  _totalRows?: number
): string => {
  if (destination && !isCurrentDestination(destination)) {
    return t('slashCmd.core.queue.changed')
  }

  // Local protection: server-tail rows cannot be removed here
  if (index < 1 || index > items.length) {
    return t('slashCmd.core.queue.invalidIndex', String(items.length))
  }

  const item = items[index - 1]

  if (!item || isProtected(item)) {
    return t('slashCmd.core.queue.protected')
  }

  try {
    removeQueue(index - 1)
  } catch (error) {
    return t('slashCmd.core.queue.notSaved', String(error))
  }

  const present = items.some(row => row === item || Boolean(item.submissionId && row.submissionId === item.submissionId))

  if (present) {
    return t('slashCmd.core.queue.kept', String(index))
  }

  refresh?.()

  return t('slashCmd.core.queue.removed', String(index), previewOf(item.display ?? item.text))
}

export const editQueueItem = (
  index: number,
  rawText: string,
  rawDisplay: string | undefined,
  items: QueueItem[],
  refresh?: () => void,
  expand?: (s: string) => string,
  attachments?: QueueItem['attachments'],
  destination?: SubmissionDestination,
  totalRows?: number
): string => {
  if (destination && !isCurrentDestination(destination)) {
    return t('slashCmd.core.queue.changed')
  }

  const rowCount = totalRows ?? items.length

  if (index < 1 || index > rowCount) {
    return t('slashCmd.core.queue.invalidIndex', String(items.length))
  }

  if (index > items.length) {
    return t('slashCmd.core.queue.gatewayFifo')
  }

  const item = items[index - 1]

  if (!item || isProtected(item)) {
    return t('slashCmd.core.queue.protected')
  }

  const text = expand ? expand(rawText) : rawText
  const display = rawDisplay ?? (rawText === text ? text : rawText)
  const previous = { attachments: item.attachments, display: item.display, text: item.text }

  item.text = text
  item.display = display

  if (attachments && attachments.length > 0) {
    item.attachments = attachments
  }

  try {
    savePendingInput(item)
  } catch (error) {
    item.text = previous.text
    item.display = previous.display
    item.attachments = previous.attachments

    return t('slashCmd.core.queue.notSaved', String(error))
  }

  refresh?.()

  return t('slashCmd.core.queue.updated', String(index), previewOf(rawText))
}

export const moveQueueItem = (
  from: number,
  to: number,
  items: QueueItem[],
  refresh?: () => void,
  destination?: SubmissionDestination,
  totalRows?: number
): string => {
  if (destination && !isCurrentDestination(destination)) {
    return t('slashCmd.core.queue.changed')
  }

  const rowCount = totalRows ?? items.length

  if (from < 1 || from > rowCount || to < 1 || to > rowCount) {
    return t('slashCmd.core.queue.moveOutOfRange', String(items.length))
  }

  if (from > items.length || to > items.length) {
    return t('slashCmd.core.queue.gatewayFifo')
  }

  const low = Math.min(from, to) - 1
  const high = Math.max(from, to) - 1

  // Middle-row protection: moving must not reorder across protected rows
  const hasProtected = items.slice(low, high + 1).some(row => !row || isProtected(row))

  if (hasProtected) {
    return t('slashCmd.core.queue.protected')
  }

  if (from !== to) {
    const backup = items.map(item => ({ createdAt: item.createdAt, item }))
    const [moved] = items.splice(from - 1, 1)
    items.splice(to - 1, 0, moved!)
    const base = Date.now()

    try {
      for (let i = 0; i < items.length; i++) {
        items[i]!.createdAt = base + i
        savePendingInput(items[i]!)
      }
    } catch (error) {
      // Failed persistence rolls back array order and journal timestamps
      backup.forEach(({ createdAt, item }, i) => {
        item.createdAt = createdAt
        items[i] = item
      })

      for (const { item } of backup) {
        try {
          savePendingInput(item)
        } catch (rollbackError) {
          return t('slashCmd.core.queue.notSaved', `${String(error)}; rollback failed: ${String(rollbackError)}`)
        }
      }

      return t('slashCmd.core.queue.notSaved', String(error))
    }
  }

  refresh?.()

  return t('slashCmd.core.queue.moved', String(from), String(to))
}

export const clearQueue = (
  items: QueueItem[],
  removeQueue: (i: number) => void,
  refresh?: () => void,
  destination?: SubmissionDestination
): string[] => {
  const current = () =>
    (!destination || isCurrentDestination(destination)) &&
    items.every(item => !item.destination || isCurrentDestination(item.destination))

  if (!current()) {
    return [t('slashCmd.core.queue.changed')]
  }

  const snapshot = [...items]

  const present = (item: QueueItem) =>
    items.some(row => row === item || Boolean(item.submissionId && row.submissionId === item.submissionId))

  let removed = 0
  let stoppedReason: string | null = null

  // Local protection: clear descending, only touching local items (never server rows)
  for (let index = snapshot.length - 1; index >= 0; index--) {
    const item = snapshot[index]!

    if (!current() || items[index] !== item) {
      stoppedReason = t('slashCmd.core.queue.changed')

      break
    }

    if (isProtected(item)) {
      continue
    }

    try {
      removeQueue(index)
    } catch (error) {
      stoppedReason = t('slashCmd.core.queue.notSaved', String(error))

      break
    }

    if (!present(item)) {
      removed++
    }
  }

  refresh?.()

  const summary = t('slashCmd.core.queue.clearSummary', String(removed), String(snapshot.length - removed))

  return stoppedReason ? [stoppedReason, summary] : [summary]
}

/** Apply a management plan to the live composer queue. Enqueue stays with the submit path. */
export function applyQueuePlan(plan: Exclude<QueuePlan, { kind: 'enqueue' }>, surface: QueueSurface): string {
  switch (plan.kind) {
    case 'usage':
      return t(USAGE_KEY[plan.key])

    case 'list':
      return listQueue(surface.items)

    case 'clear':
      return clearQueue(surface.items, surface.removeAt, surface.refresh).join('\n')

    case 'remove':
      return removeQueueItem(plan.index, surface.items, surface.removeAt, surface.refresh, undefined, surface.rows.length)

    case 'edit':
      return editQueueItem(plan.index, plan.text, plan.display, surface.items, surface.refresh, undefined, undefined, undefined, surface.rows.length)

    case 'move':
      return moveQueueItem(plan.from, plan.to, surface.items, surface.refresh, undefined, surface.rows.length)
  }
}

/** Classic CLI `/queue` slash command handler in Ink composer context. */
export function runQueueSlashCommand(arg: string, ctx: SlashRunCtx, _cmd?: string): void {
  const command = parseQueueCommand(arg)
  const sys = ctx.transcript.sys

  if (command.kind === 'prompt') {
    const text = ctx.expand ? ctx.expand(arg) : arg
    const destination = captureDestination()
    ctx.composer.enqueue(text, arg, destination, ctx.attachments ?? [])
    ctx.composer.refresh?.()

    return sys(t('slashCmd.core.queue.queued', previewOf(arg)))
  }

  if (command.kind === 'error') {
    return sys(t('slashCmd.core.queue.usage'))
  }

  const items = ctx.composer.queueRef.current
  const destination = captureDestination()
  const refresh = ctx.composer.refresh ? () => ctx.composer.refresh!() : undefined
  const removeQueue = (i: number) => ctx.composer.removeQueue(i)

  if (command.kind === 'list') {
    return sys(listQueue(items, destination))
  }

  if (command.kind === 'remove') {
    return sys(removeQueueItem(command.index, items, removeQueue, refresh, destination))
  }

  if (command.kind === 'edit') {
    return sys(
      editQueueItem(
        command.index,
        command.text,
        command.display,
        items,
        refresh,
        ctx.expand,
        ctx.attachments,
        destination
      )
    )
  }

  if (command.kind === 'move') {
    return sys(moveQueueItem(command.from, command.to, items, refresh, destination))
  }

  if (command.kind === 'add') {
    const text = ctx.expand ? ctx.expand(command.text) : command.text
    ctx.composer.enqueue(text, command.text, destination, ctx.attachments ?? [])
    refresh?.()

    return sys(t('slashCmd.core.queue.queued', previewOf(command.text)))
  }

  const lines = clearQueue(items, removeQueue, refresh, destination)
  lines.forEach(line => sys(line))
}
