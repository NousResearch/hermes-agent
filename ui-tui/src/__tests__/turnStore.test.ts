import { beforeEach, describe, expect, it } from 'vitest'

import {
  archiveDoneTodos,
  archiveTodosAtTurnEnd,
  getTurnState,
  patchTurnState,
  resetTurnState,
  toggleTodoCollapsed
} from '../app/turnStore.js'

describe('turnStore live progress helpers', () => {
  beforeEach(() => resetTurnState())

  it('archives completed todos into a transcript trail and clears the live anchor', () => {
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
    expect(getTurnState().todos).toEqual([])
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
    expect(getTurnState().todos).toEqual([])
  })

  it('returns nothing when there are no todos at turn end', () => {
    expect(archiveTodosAtTurnEnd()).toEqual([])
    expect(archiveDoneTodos()).toEqual([])
  })

  it('tracks collapsed state independently of todo content', () => {
    toggleTodoCollapsed()
    expect(getTurnState().todoCollapsed).toBe(true)

    toggleTodoCollapsed()
    expect(getTurnState().todoCollapsed).toBe(false)
  })
})
