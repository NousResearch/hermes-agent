import { describe, expect, it, vi } from 'vitest'

import { startComposerMouseSelection } from '../components/appLayout.js'
import type { TextInputMouseApi } from '../components/textInput.js'

describe('startComposerMouseSelection', () => {
  it('starts at the clicked prompt-row offset instead of forcing column 0', () => {
    const startAt = vi.fn()
    const stopImmediatePropagation = vi.fn()
    const api: TextInputMouseApi = { dragAt: vi.fn(), end: vi.fn(), startAt }

    startComposerMouseSelection(
      api,
      { button: 0, localCol: 11, localRow: 2, stopImmediatePropagation },
      4
    )

    expect(stopImmediatePropagation).toHaveBeenCalledOnce()
    expect(startAt).toHaveBeenCalledWith(2, 7)
  })

  it('pins spacer clicks to row 0 while still honoring the horizontal offset', () => {
    const startAt = vi.fn()
    const api: TextInputMouseApi = { dragAt: vi.fn(), end: vi.fn(), startAt }

    startComposerMouseSelection(api, { button: 0, localCol: 9, localRow: 5 }, 4, true)

    expect(startAt).toHaveBeenCalledWith(0, 5)
  })

  it('ignores non-left clicks', () => {
    const startAt = vi.fn()
    const api: TextInputMouseApi = { dragAt: vi.fn(), end: vi.fn(), startAt }

    startComposerMouseSelection(api, { button: 2, localCol: 9, localRow: 1 }, 4)

    expect(startAt).not.toHaveBeenCalled()
  })
})
