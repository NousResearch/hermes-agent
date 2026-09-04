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

// A revision watermark (mirroring apps/desktop/src/store/todos.ts's
// acceptRevision) must reject an out-of-order snapshot instead of letting a
// late-arriving stale event undo a newer one already on screen.
describe('turnController.recordTodos — revision watermark', () => {
  beforeEach(() => {
    resetTurnState()
    turnController.fullReset()
  })

  it('applies an unrevisioned update (tool.start-shaped patch) unconditionally', () => {
    turnController.recordTodos([{ content: 'x', id: 'a', status: 'pending' }])

    expect(getTurnState().todos).toEqual([{ content: 'x', id: 'a', status: 'pending' }])
  })

  it('applies increasing revisions in order', () => {
    turnController.recordTodos([{ content: 'x', id: 'a', status: 'pending' }], 1)
    turnController.recordTodos([{ content: 'x', id: 'a', status: 'completed' }], 2)

    expect(getTurnState().todos).toEqual([{ content: 'x', id: 'a', status: 'completed' }])
    expect(getTurnState().todoRevision).toBe(2)
  })

  it('rejects a stale (lower) revision instead of undoing the newer state', () => {
    turnController.recordTodos([{ content: 'x', id: 'a', status: 'completed' }], 5)
    turnController.recordTodos([{ content: 'x', id: 'a', status: 'pending' }], 3)

    expect(getTurnState().todos).toEqual([{ content: 'x', id: 'a', status: 'completed' }])
    expect(getTurnState().todoRevision).toBe(5)
  })
})

// applyTodoSnapshot() backs both the `todo.updated` gateway event and
// session.resume/session.activate's `todo_state` restore.
describe('turnController.applyTodoSnapshot', () => {
  beforeEach(() => {
    resetTurnState()
    turnController.fullReset()
  })

  it('applies a live (running) snapshot even if the list is still active', () => {
    turnController.applyTodoSnapshot([{ content: 'x', id: 'a', status: 'in_progress' }], 1, true)

    expect(getTurnState().todos).toEqual([{ content: 'x', id: 'a', status: 'in_progress' }])
  })

  it('restores a finished list on resume of an idle session', () => {
    turnController.applyTodoSnapshot([{ content: 'x', id: 'a', status: 'completed' }], 1, false)

    expect(getTurnState().todos).toEqual([{ content: 'x', id: 'a', status: 'completed' }])
  })

  it('drops a still-active list on resume of an idle session instead of re-pinning a stuck panel', () => {
    turnController.applyTodoSnapshot([{ content: 'x', id: 'a', status: 'in_progress' }], 1, false)

    expect(getTurnState().todos).toEqual([])
  })

  it('ignores an unused-store snapshot (empty todos, revision 0)', () => {
    turnController.recordTodos([{ content: 'x', id: 'a', status: 'pending' }])
    turnController.applyTodoSnapshot([], 0, false)

    expect(getTurnState().todos).toEqual([{ content: 'x', id: 'a', status: 'pending' }])
  })
})
