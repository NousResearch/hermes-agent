import { describe, expect, it, vi } from 'vitest'

import { copySelectionText } from './copy-selection.js'

describe('selection clipboard receipt', () => {
  it('announces a successful copy after writing its terminal sequence', async () => {
    const events: string[] = []
    const copy = vi.fn(async () => ({ sequence: 'osc52', success: true }))

    const result = await copySelectionText(
      'hello',
      sequence => events.push(sequence),
      [text => events.push(text)],
      copy
    )

    expect(result).toBe('hello')
    expect(events).toEqual(['osc52', 'hello'])
  })

  it('does not report copied text when all clipboard paths failed', async () => {
    const receipt = vi.fn()
    const copy = vi.fn(async () => ({ sequence: '', success: false }))
    expect(await copySelectionText('hello', vi.fn(), [receipt], copy)).toBe('')
    expect(receipt).not.toHaveBeenCalled()
  })
})
