import { describe, expect, it } from 'vitest'

import { statusModel } from '../components/appLayout.js'

describe('statusModel', () => {
  it('uses the atomic runtime model when present', () => {
    expect(statusModel('claude-fallback', 'gpt-primary')).toBe('claude-fallback')
  })

  it.each(['', '   ', undefined])('falls back to session info for an absent runtime model: %j', model => {
    expect(statusModel(model, 'gpt-primary')).toBe('gpt-primary')
  })
})
