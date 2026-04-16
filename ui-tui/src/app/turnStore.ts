import { atom } from 'nanostores'

import type { ActiveTool, ActivityItem, SubagentProgress } from '../types.js'

export interface TurnState {
  activity: ActivityItem[]
  reasoning: string
  reasoningActive: boolean
  reasoningStreaming: boolean
  reasoningTokens: number
  streaming: string
  subagents: SubagentProgress[]
  toolTokens: number
  tools: ActiveTool[]
  turnTrail: string[]
}

function buildTurnState(): TurnState {
  return {
    activity: [],
    reasoning: '',
    reasoningActive: false,
    reasoningStreaming: false,
    reasoningTokens: 0,
    streaming: '',
    subagents: [],
    toolTokens: 0,
    tools: [],
    turnTrail: []
  }
}

export const $turnState = atom<TurnState>(buildTurnState())

export const getTurnState = () => $turnState.get()

export const patchTurnState = (next: Partial<TurnState> | ((state: TurnState) => TurnState)) => {
  if (typeof next === 'function') {
    $turnState.set(next($turnState.get()))

    return
  }

  $turnState.set({ ...$turnState.get(), ...next })
}

export const resetTurnState = () => $turnState.set(buildTurnState())
