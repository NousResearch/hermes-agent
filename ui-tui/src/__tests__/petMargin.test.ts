import { describe, expect, it } from 'vitest'

import { petMarginRight } from '../app/usePet.js'

describe('petMarginRight', () => {
  it('defaults to one for older gateways', () => {
    expect(petMarginRight(null, {})).toBe(1)
  })

  it('updates cached frames from metadata while preserving zero', () => {
    expect(petMarginRight(null, { marginRight: 0 })).toBe(0)
    expect(petMarginRight(null, { marginRight: 7 })).toBe(7)
  })

  it('prefers the later frame response when configuration changes between calls', () => {
    expect(petMarginRight({ marginRight: 0 }, { marginRight: 6 })).toBe(0)
  })
})
