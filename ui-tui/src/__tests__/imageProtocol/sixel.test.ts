import { describe, it, expect } from 'vitest'
import { encodeSixel } from '../../lib/imageProtocol/sixel.js'

describe('encodeSixel', () => {
  it('wraps a payload in DCS/ST', () => {
    const result = encodeSixel({ payload: '?###~' })
    expect(result).toMatch(/^\x1bP0;0;0;0q/)
    expect(result).toMatch(/\x1b\\$/)
  })
})
