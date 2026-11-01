import { describe, expect, it } from 'vitest'

import { atHistoryBoundary, setInputSelection } from '../app/inputSelectionStore.js'
import { cursorLayout, visualLineNav } from '../lib/inputMetrics.js'

describe('visual cursor navigation', () => {
  it('keeps a wrapped row out of history and lets users disable edge history', () => {
    setInputSelection({
      canMoveUp: true,
      canMoveDown: false,
      value: 'wrapped draft',
      start: 10,
      end: 10,
      clear() {},
      collapseToEnd() {},
      copy() {},
      cut() {}
    })
    expect(atHistoryBoundary('wrapped draft', -1, 'auto')).toBe(false)
    expect(atHistoryBoundary('wrapped draft', 1, 'auto')).toBe(true)
    expect(atHistoryBoundary('wrapped draft', 1, 'cursor')).toBe(false)
    setInputSelection(null)
  })
  it('moves within a wrapped paragraph before reaching history', () => {
    const value = 'alpha beta gamma delta epsilon'
    const position = value.length
    const before = cursorLayout(value, position, 12)
    const target = visualLineNav(value, position, -1, 12)
    expect(target).not.toBeNull()
    expect(cursorLayout(value, target!, 12).line).toBe(before.line - 1)
    expect(visualLineNav(value, 0, -1, 12)).toBeNull()
  })

  it('does not split an emoji or a combining character while moving between rows', () => {
    const value = '🧪e\u0301abc\n🧪e\u0301xyz'
    const target = visualLineNav(value, value.length, -1, 20)
    expect(target).toBe(7)
    expect(visualLineNav(value, target!, 1, 20)).toBe(value.length)
  })
})
