import { describe, expect, it } from 'vitest'

import { quotaLineIsCompact } from '../components/branding.js'

const window = { label: 'Weekly', remainingPercent: 81, resetIn: '5d 0h', usedPercent: 19 }

describe('quotaLineIsCompact', () => {
  it('drops to the compact form in the fixed hero track', () => {
    // `Weekly: 81% left (19% used) · resets in 5d 0h` is 44 cols; the hero
    // column is leftW (~32), where the verbose form loses its countdown.
    expect(quotaLineIsCompact(window, 32)).toBe(true)
  })

  it('keeps the verbose form when the column is wide enough', () => {
    expect(quotaLineIsCompact(window, 60)).toBe(false)
  })

  it('measures without the reset clause when the provider reports no reset', () => {
    expect(quotaLineIsCompact({ ...window, resetIn: '' }, 30)).toBe(false)
  })
})
