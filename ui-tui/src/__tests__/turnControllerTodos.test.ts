import { beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'

// turnController.recordTodos() parses the raw `todo` tool payload into
// TodoItem[]. Nested subtasks (apps/desktop's `parent` field) must survive
// this parse — the TUI todo panel renders hierarchy from it via todoTree().
describe('turnController.recordTodos — preserves the parent field', () => {
  beforeEach(() => {
    resetTurnState()
    turnController.fullReset()
  })

  it('keeps parent on a valid nested subtask', () => {
    turnController.recordTodos([
      { content: 'Ship feature', id: 'wp1', status: 'in_progress' },
      { content: 'Write tests', id: 't1', parent: 'wp1', status: 'pending' }
    ])

    expect(getTurnState().todos).toEqual([
      { content: 'Ship feature', id: 'wp1', status: 'in_progress' },
      { content: 'Write tests', id: 't1', parent: 'wp1', status: 'pending' }
    ])
  })

  it('drops a self-referential parent instead of keeping a self-loop', () => {
    turnController.recordTodos([{ content: 'x', id: 'a', parent: 'a', status: 'pending' }])

    expect(getTurnState().todos).toEqual([{ content: 'x', id: 'a', status: 'pending' }])
  })

  it('omits parent entirely when absent, matching pre-nesting payloads', () => {
    turnController.recordTodos([{ content: 'x', id: 'a', status: 'pending' }])

    expect(getTurnState().todos).toEqual([{ content: 'x', id: 'a', status: 'pending' }])
  })
})
