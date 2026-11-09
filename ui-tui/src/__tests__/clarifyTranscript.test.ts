import { describe, expect, it } from 'vitest'

import { clarifySettledRows } from '../app/clarifyTranscript.js'

describe('clarifySettledRows', () => {
  it("writes main's `question → answer` bubble and the `1 question` tool row for one card (row 8)", () => {
    const rows = clarifySettledRows([{ qid: 'p1', question: 'Pick a colour?' }], { p1: 'red (Recommended)' })

    expect(rows.userText).toBe('Pick a colour? → red (Recommended)')
    expect(rows.trail).toContain('1 question')
    expect(rows.trail).not.toContain('Pick a colour?')
  })

  it('reads a blank answer as skipped', () => {
    expect(clarifySettledRows([{ qid: 'p1', question: 'Name?' }], { p1: ' ' }).userText).toMatch(/^Name\? → /)
  })
})
