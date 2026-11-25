import { describe, expect, it } from 'vitest'

import { clarifySettledRows } from '../app/clarifyTranscript.js'
import { toTranscriptMessages } from '../domain/messages.js'

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

  it('shows the stored answers again when the session is resumed', () => {
    const content = JSON.stringify({
      outcome: 'submitted',
      responses: [
        { question: 'Pick a colour?', status: 'answered', user_response: 'red' },
        { question: 'Sizes?', status: 'answered', user_response: ['S', 'M'] }
      ]
    })

    const rows = toTranscriptMessages([
      { role: 'user', text: 'paint it' },
      { content, context: '2 questions', name: 'clarify', role: 'tool' },
      { role: 'assistant', text: 'done' }
    ])

    expect(rows.map(row => [row.role, row.text])).toEqual([
      ['user', 'paint it'],
      ['system', ''],
      ['user', 'Pick a colour? → red\nSizes? → S, M'],
      ['assistant', 'done']
    ])
  })

  it.each([
    ['cancelled', '(cancelled)'],
    ['timed_out', '(timed out)']
  ])('shows a %s form as the abandoned-form line it showed live, not as skips', (outcome, reason) => {
    const content = JSON.stringify({
      outcome,
      responses: [
        { question: 'Pick a colour?', status: 'answered', user_response: 'red' },
        { question: 'Sizes?', status: 'unanswered', user_response: null }
      ]
    })

    const [, line] = toTranscriptMessages([{ content, context: '2 questions', name: 'clarify', role: 'tool' }])

    expect(line?.role).toBe('system')
    expect(line?.text).toContain('✓ Pick a colour? → red')
    expect(line?.text).toContain('· Sizes? (no answer)')
    expect(line?.text).toContain(reason)
    expect(line?.text).not.toContain('(skipped)')
  })
})
