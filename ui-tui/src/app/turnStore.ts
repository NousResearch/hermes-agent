import { atom } from 'nanostores'
import { useSyncExternalStore } from 'react'

import { isTodoDone } from '../lib/liveProgress.js'
import type { ActiveTool, ActivityItem, Msg, SubagentProgress, TodoItem } from '../types.js'

const buildTurnState = (): TurnState => ({
  activity: [],
  outcome: '',
  reasoning: '',
  reasoningActive: false,
  reasoningStreaming: false,
  reasoningTokens: 0,
  streamPendingTools: [],
  streamSegments: [],
  streaming: '',
  subagents: [],
  todoArchiveSignature: '',
  todoCollapsed: false,
  todos: [],
  toolTokens: 0,
  tools: [],
  turnTrail: []
})

export const $turnState = atom<TurnState>(buildTurnState())

export const getTurnState = () => $turnState.get()

const subscribeTurn = (cb: () => void) => $turnState.listen(() => cb())

export const useTurnSelector = <T>(selector: (state: TurnState) => T): T =>
  useSyncExternalStore(
    subscribeTurn,
    () => selector($turnState.get()),
    () => selector($turnState.get())
  )

export const patchTurnState = (next: Partial<TurnState> | ((state: TurnState) => TurnState)) =>
  $turnState.set(typeof next === 'function' ? next($turnState.get()) : { ...$turnState.get(), ...next })

export const toggleTodoCollapsed = () => patchTurnState(state => ({ ...state, todoCollapsed: !state.todoCollapsed }))

export const todoSignature = (todos: readonly TodoItem[]) =>
  JSON.stringify(todos.map(todo => [todo.id, todo.content, todo.status]))

export const archiveDoneTodos = () => archiveTodosAtTurnEnd()

export const archiveTodosAtTurnEnd = () => {
  const state = $turnState.get()

  if (!state.todos.length) {
    return []
  }

  const done = isTodoDone(state.todos)
  const signature = todoSignature(state.todos)

  if (!done && state.todoArchiveSignature === signature) {
    return []
  }

  const msg: Msg = {
    kind: 'trail',
    role: 'system',
    text: '',
    todos: state.todos,
    ...(done ? { todoCollapsedByDefault: true } : { todoIncomplete: true })
  }

  if (done) {
    patchTurnState({ todoArchiveSignature: '', todoCollapsed: false, todos: [] })
  } else {
    // Keep unfinished work anchored above the composer after a turn stops.
    // The archived copy preserves scrollback; the signature prevents the same
    // unchanged snapshot being appended again on every follow-up turn.
    patchTurnState({ todoArchiveSignature: signature, todoCollapsed: false })
  }

  return [msg]
}

export const resetTurnState = () => $turnState.set(buildTurnState())

export interface TurnState {
  activity: ActivityItem[]
  outcome: string
  reasoning: string
  reasoningActive: boolean
  reasoningStreaming: boolean
  reasoningTokens: number
  streamPendingTools: string[]
  streamSegments: Msg[]
  streaming: string
  subagents: SubagentProgress[]
  todoArchiveSignature: string
  todoCollapsed: boolean
  todos: TodoItem[]
  toolTokens: number
  tools: ActiveTool[]
  turnTrail: string[]
}
