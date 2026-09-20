import { describe, expect, it } from 'vitest'

import { shouldSetVirtualClamp } from '../hooks/useVirtualHistory.js'

describe('virtual history clamp bounds', () => {
  it('does not clamp sticky live tail content', () => {
    expect(shouldSetVirtualClamp({ itemCount: 20, sticky: true, viewportHeight: 10 })).toBe(false)
  })

  it('sets clamp bounds after manual scroll breaks sticky mode', () => {
    expect(shouldSetVirtualClamp({ itemCount: 20, sticky: false, viewportHeight: 10 })).toBe(true)
  })

  it('does not clamp while a live tail is growing below virtual history', () => {
    expect(shouldSetVirtualClamp({ itemCount: 20, liveTailActive: true, sticky: false, viewportHeight: 10 })).toBe(
      false
    )
  })
})
