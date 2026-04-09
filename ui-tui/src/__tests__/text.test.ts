import { describe, expect, it } from 'vitest'

import { sameToolTrailGroup } from '../lib/text.js'

describe('sameToolTrailGroup', () => {
  it('matches bare check lines', () => {
    expect(sameToolTrailGroup('🔍 searching', '🔍 searching ✓')).toBe(true)
    expect(sameToolTrailGroup('🔍 searching', '🔍 searching ✗')).toBe(true)
  })

  it('matches contextual lines', () => {
    expect(sameToolTrailGroup('🔍 searching', '🔍 searching: * ✓')).toBe(true)
    expect(sameToolTrailGroup('🔍 searching', '🔍 searching: foo ✓')).toBe(true)
  })

  it('rejects other tools', () => {
    expect(sameToolTrailGroup('🔍 searching', '📖 reading ✓')).toBe(false)
    expect(sameToolTrailGroup('🔍 searching', '🔍 searching extra ✓')).toBe(false)
  })
})
