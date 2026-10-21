import { describe, expect, it } from 'vitest'

import { sessionLabel } from './sessionLabel.js'

describe('sessionLabel', () => {
  it("shows a canonical durable id as main's 8-character label", () => {
    expect(sessionLabel(`local-${'ab12cd34'}${'0'.repeat(56)}`)).toBe('ab12cd34')
  })

  it('leaves an 8-character label as it is', () => {
    expect(sessionLabel('a707d4c3')).toBe('a707d4c3')
    expect(sessionLabel(null)).toBeNull()
  })

  // PR #106742 row 33: main's banner named a resumed session by its 8-hex runtime label, never by the
  // stored `YYYYMMDD_HHMMSS_xxxxxx` id the canonical owner keys it on.
  it('shows a resumed stored id as a stable 8-character label', () => {
    const label = sessionLabel('20261012_134501_abcdef')
    expect(label).toMatch(/^[0-9a-f]{8}$/)
    expect(sessionLabel('20261012_134501_abcdef')).toBe(label)
    expect(sessionLabel('20261012_134501_abcdeg')).not.toBe(label)
  })
})
