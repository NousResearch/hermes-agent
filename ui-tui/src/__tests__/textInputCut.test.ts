import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $clipboardFeedback } from '../app/clipboardFeedbackStore.js'
import { cutSelection } from '../lib/inputClipboard.js'

beforeEach(() => {
  vi.useFakeTimers()
  $clipboardFeedback.set(null)
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('cutSelection (transactional cut)', () => {
  it('removes the selection only after the clipboard write succeeds', async () => {
    const write = vi.fn().mockResolvedValue(true)
    const removeSelection = vi.fn()

    const ok = await cutSelection('hello', write, removeSelection)

    expect(ok).toBe(true)
    expect(write).toHaveBeenCalledWith('hello')
    expect(removeSelection).toHaveBeenCalledOnce()
    expect($clipboardFeedback.get()?.text).toBe('cut 5 characters')
  })

  it('keeps the text intact when the clipboard write fails (headless/SSH)', async () => {
    const write = vi.fn().mockResolvedValue(false)
    const removeSelection = vi.fn()

    const ok = await cutSelection('hello', write, removeSelection)

    expect(ok).toBe(false)
    expect(write).toHaveBeenCalledWith('hello')
    // Text must NOT be removed. A failed write would otherwise destroy it with no
    // clipboard copy to paste back.
    expect(removeSelection).not.toHaveBeenCalled()
    expect($clipboardFeedback.get()).toBeNull()
  })

  it('awaits the write before removing (no fire-and-forget removal)', async () => {
    let resolveWrite: (value: boolean) => void = () => {}

    const write = vi.fn(
      () =>
        new Promise<boolean>(resolve => {
          resolveWrite = resolve
        })
    )

    const removeSelection = vi.fn()

    const pending = cutSelection('hello', write, removeSelection)

    // While the write is still pending the selection must remain untouched.
    await Promise.resolve()
    expect(removeSelection).not.toHaveBeenCalled()

    resolveWrite(true)
    await pending

    expect(removeSelection).toHaveBeenCalledOnce()
  })

  it('reports a copy if the selection changed before the clipboard write completed', async () => {
    const write = vi.fn().mockResolvedValue(true)
    await cutSelection('🧪e\u0301', write, () => false)
    expect($clipboardFeedback.get()?.text).toBe('copied 2 characters')
  })
})
