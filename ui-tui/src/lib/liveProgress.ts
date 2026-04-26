import type { Msg, TodoItem } from '../types.js'

export const isTodoDone = (todos: readonly TodoItem[]) =>
  todos.length > 0 && todos.every(todo => todo.status === 'completed' || todo.status === 'cancelled')

export const isToolShelfMessage = (msg: Msg | undefined) =>
  Boolean(msg?.kind === 'trail' && !msg.text && !msg.thinking?.trim() && msg.tools?.length)

const canHoldToolShelf = (msg: Msg | undefined) =>
  Boolean(msg?.kind === 'trail' && !msg.text && (msg.thinking?.trim() || msg.tools?.length))

export const appendToolShelfMessage = (prev: readonly Msg[], msg: Msg): Msg[] => {
  if (!isToolShelfMessage(msg)) {
    return [...prev, msg]
  }

  for (let index = prev.length - 1; index >= 0; index--) {
    const candidate = prev[index]

    if (canHoldToolShelf(candidate)) {
      const next = [...prev]

      next[index] = { ...candidate!, tools: [...(candidate!.tools ?? []), ...(msg.tools ?? [])] }

      return next
    }

    if (candidate?.kind !== 'trail' || candidate.text) {
      break
    }
  }

  return [...prev, msg]
}
