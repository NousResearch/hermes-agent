import { describe, expect, it, vi } from 'vitest'

import { handleInputSelectionClipboard } from '../app/useInputHandlers.js'

const selection = (start = 1, end = 4) => ({
  clear: vi.fn(),
  collapseToEnd: vi.fn(),
  copy: vi.fn(),
  cut: vi.fn(),
  end,
  start,
  value: 'hello'
})

describe('handleInputSelectionClipboard', () => {
  it('copies an active composer selection', () => {
    const active = selection()

    expect(handleInputSelectionClipboard(active, 'copy')).toBe(true)
    expect(active.copy).toHaveBeenCalledOnce()
    expect(active.cut).not.toHaveBeenCalled()
  })

  it('cuts an active composer selection', () => {
    const active = selection()

    expect(handleInputSelectionClipboard(active, 'cut')).toBe(true)
    expect(active.cut).toHaveBeenCalledOnce()
    expect(active.copy).not.toHaveBeenCalled()
  })

  it('leaves shortcuts available when there is no active selection', () => {
    expect(handleInputSelectionClipboard(null, 'copy')).toBe(false)
    expect(handleInputSelectionClipboard(selection(2, 2), 'cut')).toBe(false)
  })
})
