import { beforeEach, describe, expect, it } from 'vitest'

import {
  appendTurnSegment,
  archiveDoneTodos,
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

    expect(archiveDoneTodos()).toEqual([
      {
        kind: 'trail',
        role: 'system',
        text: '',
        todos: [
          { content: 'prep', id: 'prep', status: 'completed' },
          { content: 'serve', id: 'serve', status: 'completed' }
        ]
      }
    ])
    expect(getTurnState().todos).toEqual([])
  })

  it('does not archive active todos', () => {
    patchTurnState({ todos: [{ content: 'cook', id: 'cook', status: 'in_progress' }] })

    expect(archiveDoneTodos()).toEqual([])
    expect(getTurnState().todos).toHaveLength(1)
  })

  it('tracks collapsed state independently of todo content', () => {
    toggleTodoCollapsed()
    expect(getTurnState().todoCollapsed).toBe(true)

    toggleTodoCollapsed()
    expect(getTurnState().todoCollapsed).toBe(false)
  })

  it('merges adjacent live tool shelves before rendering', () => {
    appendTurnSegment({ kind: 'trail', role: 'system', text: '', tools: ['one ✓'] })
    appendTurnSegment({ kind: 'trail', role: 'system', text: '', tools: ['two ✓'] })

    expect(getTurnState().streamSegments).toEqual([
      { kind: 'trail', role: 'system', text: '', tools: ['one ✓', 'two ✓'] }
    ])
  })
})
