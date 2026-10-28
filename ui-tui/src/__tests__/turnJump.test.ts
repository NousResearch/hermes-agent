import { describe, expect, it, vi } from 'vitest'

import { jumpToTurn } from '../domain/viewport.js'

const handle = () => ({
  adjustScrollTop: vi.fn(),
  clampScrollTopToChildren: vi.fn(),
  scrollTo: vi.fn(),
  scrollToBottom: vi.fn(),
  scrollToElement: vi.fn(),
  scrollTop: 0
})

const offsets = [0, 5, 12, 20, 27] // 5 messages; users at 0, 12, 27
const users = [0, 2, 4]

describe('jumpToTurn', () => {
  it('prev from turn 2 lands on the previous user message offset', () => {
    const h = handle()
    const jumped = jumpToTurn(h as never, offsets, users, 4, -1)

    expect(jumped).toBe(true)
    expect(h.scrollTo).toHaveBeenCalledWith(offsets[2] ?? 0)
  })

  it('prev at the top turn does not jump', () => {
    const h = handle()
    expect(jumpToTurn(h as never, offsets, users, 0, -1)).toBe(false)
    expect(h.scrollTo).not.toHaveBeenCalled()
  })

  it('next on the last turn scrolls to bottom (back to live)', () => {
    const h = handle()
    const jumped = jumpToTurn(h as never, offsets, users, 4, 1)

    expect(jumped).toBe(true)
    expect(h.scrollToBottom).toHaveBeenCalled()
  })

  it('next from turn 0 lands on the next user message offset', () => {
    const h = handle()
    const jumped = jumpToTurn(h as never, offsets, users, 0, 1)

    expect(jumped).toBe(true)
    expect(h.scrollTo).toHaveBeenCalledWith(offsets[2] ?? 0)
  })

  it('returns false with no user messages', () => {
    const h = handle()
    expect(jumpToTurn(h as never, offsets, [], 0, 1)).toBe(false)
  })

  it('clamps a stale current index into range', () => {
    const h = handle()
    const jumped = jumpToTurn(h as never, offsets, users, 99, -1)

    expect(jumped).toBe(true)
    expect(h.scrollTo).toHaveBeenCalledWith(offsets[2] ?? 0)
  })
})

describe('currentTurnIndex', () => {
  it('returns the last user message whose block starts at or above the scroll top', async () => {
    const { currentTurnIndex } = await import('../domain/viewport.js')

    // offsets[0]=0, offsets[2]=12, offsets[4]=27 (same layout as jumpToTurn tests)
    expect(currentTurnIndex([0, 2, 4], [0, 5, 12, 20, 27], 13)).toBe(2)
    expect(currentTurnIndex([0, 2, 4], [0, 5, 12, 20, 27], 26)).toBe(2)
    expect(currentTurnIndex([0, 2, 4], [0, 5, 12, 20, 27], 1)).toBe(0)
    expect(currentTurnIndex([0, 2, 4], [0, 5, 12, 20, 27], 0)).toBe(0)
    expect(currentTurnIndex([0, 2, 4], [0, 5, 12, 20, 27], 99)).toBe(4)
    expect(currentTurnIndex([], [0, 5], 5)).toBe(-1)
  })
})
