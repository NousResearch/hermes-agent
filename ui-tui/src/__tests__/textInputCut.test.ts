import { describe, expect, it, vi } from 'vitest'

import { cutSelection } from '../components/textInput.js'

describe('cutSelection (transactional cut)', () => {
  it('removes the selection only after the clipboard write succeeds', async () => {
    const write = vi.fn().mockResolvedValue(true)
    const removeSelection = vi.fn()

    const ok = await cutSelection('hello', write, removeSelection)

    expect(ok).toBe(true)
    expect(write).toHaveBeenCalledWith('hello')
    expect(removeSelection).toHaveBeenCalledOnce()
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
})
