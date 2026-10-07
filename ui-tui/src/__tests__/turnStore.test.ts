import { beforeEach, describe, expect, it } from 'vitest'

import {
  archiveDoneTodos,
  archiveTodosAtTurnEnd,
  getTurnState,
  patchTurnState,
  resetTurnState
} from '../app/turnStore.js'

describe('turnStore live progress helpers', () => {
  beforeEach(() => resetTurnState())

  it('archives completed todos into a transcript trail and keeps the live panel', () => {
    patchTurnState({
      todos: [
        { content: 'prep', id: 'prep', status: 'completed' },
        { content: 'serve', id: 'serve', status: 'completed' }
      ]
    })

    expect(archiveTodosAtTurnEnd()).toEqual([
      {
        kind: 'trail',
        role: 'system',
        text: '',
        todoCollapsedByDefault: true,
        todos: [
          { content: 'prep', id: 'prep', status: 'completed' },
          { content: 'serve', id: 'serve', status: 'completed' }
        ]
      }
    ])
    // The live panel persists across turns so it stays visible between the
    // transcript and the composer; the trail message alone carries history.
    expect(getTurnState().todos).toEqual([
      { content: 'prep', id: 'prep', status: 'completed' },
      { content: 'serve', id: 'serve', status: 'completed' }
    ])
  })

  it('archives incomplete todos with an incomplete flag so the hint renders', () => {
    patchTurnState({
      todos: [
        { content: 'cook', id: 'cook', status: 'completed' },
        { content: 'serve', id: 'serve', status: 'in_progress' },
        { content: 'eat', id: 'eat', status: 'pending' }
      ]
    })

    const archived = archiveTodosAtTurnEnd()
    expect(archived).toHaveLength(1)
    expect(archived[0]!.todoIncomplete).toBe(true)
    expect(archived[0]!.todos?.map(t => t.id)).toEqual(['cook', 'serve', 'eat'])
    expect(getTurnState().todos.map(t => t.id)).toEqual(['cook', 'serve', 'eat'])
  })

  it('returns nothing when there are no todos at turn end', () => {
    expect(archiveTodosAtTurnEnd()).toEqual([])
    expect(archiveDoneTodos()).toEqual([])
  })
})
