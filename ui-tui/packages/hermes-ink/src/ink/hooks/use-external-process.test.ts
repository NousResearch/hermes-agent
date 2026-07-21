/**
 * Regression test for #20640 — after withInkSuspended returns, the terminal
 * focus state must be reset to 'unknown' so the first repaint shows the
 * cursor even when a focus-lost event fired while the editor owned the TTY.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

// Stub the Ink instance registry so we don't need a real Ink render.
vi.mock('../instances.js', () => ({
  default: new Map()
}))

// Capture calls to resetTerminalFocusState so we can assert it's called.
const resetTerminalFocusState = vi.fn()

vi.mock('../terminal-focus-state.js', () => ({
  resetTerminalFocusState,
  setTerminalFocused: vi.fn(),
  getTerminalFocused: vi.fn(() => true),
  getTerminalFocusState: vi.fn(() => 'unknown'),
  subscribeTerminalFocus: vi.fn(() => () => {})
}))

describe('withInkSuspended', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('resets terminal focus state after the external process exits (no Ink instance)', async () => {
    // When there's no live Ink instance the function falls through to just
    // awaiting run() — resetTerminalFocusState should still be called. (#20640)
    const { withInkSuspended } = await import('./use-external-process.js')

    await withInkSuspended(async () => {})

    expect(resetTerminalFocusState).toHaveBeenCalledOnce()
  })

  it('resets terminal focus state even when the external process throws', async () => {
    // The focus reset must happen in a finally block so a crashing editor
    // does not permanently hide the cursor. (#20640)
    const { withInkSuspended } = await import('./use-external-process.js')

    await expect(
      withInkSuspended(async () => {
        throw new Error('editor crash')
      })
    ).rejects.toThrow('editor crash')

    expect(resetTerminalFocusState).toHaveBeenCalledOnce()
  })
})
