import type { QueueItem } from '../hooks/useQueue.js'
import type { SessionInfo } from '../types.js'

export type QueueCommand =
  | { kind: 'list' | 'clear' | 'error' | 'prompt' }
  | { kind: 'remove'; index: number }
  | { kind: 'edit'; index: number; text: string; display?: string }
  | { kind: 'move'; from: number; to: number }
  | { kind: 'add'; text: string }

type VerbHandler = (tail: string[], rest: string) => QueueCommand

const parseIndex = (token?: string): number | null => {
  if (!token || !/^[1-9]\d*$/.test(token)) {
    return null
  }

  const value = Number(token)

  return Number.isSafeInteger(value) ? value : null
}

const handleList: VerbHandler = tail => (tail.length ? { kind: 'error' } : { kind: 'list' })

const handleClear: VerbHandler = tail => (tail.length ? { kind: 'error' } : { kind: 'clear' })

const handleRemove: VerbHandler = tail => {
  if (tail.length !== 1) {
    return { kind: 'error' }
  }

  const index = parseIndex(tail[0])

  return index !== null ? { kind: 'remove', index } : { kind: 'error' }
}

const handleEdit: VerbHandler = (tail, rest) => {
  if (tail.length < 2) {
    return { kind: 'error' }
  }

  const index = parseIndex(tail[0])

  if (index === null) {
    return { kind: 'error' }
  }

  const text = rest.slice(tail[0]!.length).trim()

  return text ? { kind: 'edit', index, text } : { kind: 'error' }
}

const handleMove: VerbHandler = tail => {
  if (tail.length !== 2) {
    return { kind: 'error' }
  }

  const from = parseIndex(tail[0])
  const to = parseIndex(tail[1])

  return from !== null && to !== null ? { kind: 'move', from, to } : { kind: 'error' }
}

const handleAdd: VerbHandler = (tail, rest) => {
  if (tail.length === 0 || !rest) {
    return { kind: 'error' }
  }

  return { kind: 'add', text: rest }
}

const VERB_HANDLERS: Record<string, VerbHandler> = {
  add: handleAdd,
  clear: handleClear,
  del: handleRemove,
  delete: handleRemove,
  edit: handleEdit,
  list: handleList,
  ls: handleList,
  move: handleMove,
  pop: handleRemove,
  remove: handleRemove,
  rm: handleRemove,
  set: handleEdit,
  show: handleList
}

/** Only literal lowercase verbs are administrative; expanded paste contents are never reparsed. */
export function parseQueueCommand(rawArgs: string): QueueCommand {
  const args = rawArgs.trim()

  if (!args) {
    return { kind: 'list' }
  }

  const [verb, ...tail] = args.split(/\s+/)
  const handler = VERB_HANDLERS[verb!]

  if (!handler) {
    return { kind: 'prompt' }
  }

  const rest = args.slice(verb!.length).trim()

  return handler(tail, rest)
}

/** Classify only the literal parsed slash name and arguments, before paste expansion. */
export function isQueueAdministrativeCommand(name: string, args: string): boolean {
  return (name === 'queue' || name === 'q') && parseQueueCommand(args).kind !== 'prompt'
}

/** An authority receipt protects a local row even before the reconciliation effect runs. */
export function isQueueItemProtected(item: QueueItem, receipts: SessionInfo['pending_submissions']): boolean {
  return Boolean(
    item.inFlight ||
      receipts?.some(
        receipt =>
          item.submissionId === (receipt.input_id ?? receipt.admission_id) &&
          Boolean(item.destination?.storedSid) &&
          item.destination?.storedSid === receipt.target_session_id &&
          (item.destination?.ownerHome ?? item.destination?.profileHome) === receipt.target_profile_home
      )
  )
}
