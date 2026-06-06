import { describe, it, expect } from 'vitest'
import { encodeITerm } from '../../lib/imageProtocol/iterm.js'

describe('encodeITerm', () => {
  it('produces the canonical iTerm2 escape sequence', () => {
    const png = Buffer.from([0x89, 0x50, 0x4e, 0x47]) // PNG header bytes
    const result = encodeITerm(png, { width: 1, height: 1, name: 'a.png' })
    expect(result).toMatch(/^\x1b]1337;File=/)
    expect(result).toMatch(/width=1/)
    expect(result).toMatch(/height=1/)
    expect(result).toMatch(/inline=1/)
    expect(result).toMatch(/\x1b\\$/)
  })
})
