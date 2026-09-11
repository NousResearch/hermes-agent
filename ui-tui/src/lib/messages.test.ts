import { describe, expect, it } from 'vitest'

import { appendTranscriptMessage } from './messages.js'

describe('appendTranscriptMessage', () => {
  it('merges adjacent tool-only shelves into one transcript row', () => {
    const out = appendTranscriptMessage([{ kind: 'trail', role: 'system', text: '', tools: ['Terminal("one") ✓'] }], {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: ['Terminal("two") ✓']
    })

    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', tools: ['Terminal("one") ✓', 'Terminal("two") ✓'] }
    ])
  })

  it('merges tool shelves into the nearest thinking shelf', () => {
    const out = appendTranscriptMessage(
      [{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['Terminal("one") ✓'] }],
      { kind: 'trail', role: 'system', text: '', tools: ['Terminal("two") ✓'] }
    )

    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['Terminal("one") ✓', 'Terminal("two") ✓'] }
    ])
  })

  it('skips an exact adjacent replay of a plain message (#88362)', () => {
    // session re-activate replays a tail event that the transcript snapshot
    // already contained: same role+text as the previous row → drop it.
    const prev = [
      { role: 'user' as const, text: 'summarize the log' },
      { role: 'assistant' as const, text: 'done' }
    ]
    const out = appendTranscriptMessage(prev, { role: 'assistant', text: 'done' })
    expect(out).toEqual(prev)
  })

  it('keeps a same-text message when the role differs', () => {
    const out = appendTranscriptMessage([{ role: 'user', text: 'ok' }], { role: 'assistant', text: 'ok' })
    expect(out).toHaveLength(2)
  })

  it('keeps repeated plain text when a different message sits between', () => {
    // A genuine repeat later in the conversation (assistant row between the
    // two user rows) is NOT an adjacent replay — keep it.
    const out = appendTranscriptMessage(
      [
        { role: 'user', text: 'again' },
        { role: 'assistant', text: 'sure' }
      ],
      { role: 'user', text: 'again' }
    )
    expect(out).toHaveLength(3)
  })

  it('does not apply the replay guard to special-kind rows', () => {
    // kind='trail' rows legitimately repeat (tool-shelf merge path);
    // the guard must not swallow them before the merge logic runs.
    const out = appendTranscriptMessage(
      [{ kind: 'trail', role: 'system', text: '', tools: ['Terminal("one") ✓'] }],
      { kind: 'trail', role: 'system', text: '', tools: ['Terminal("two") ✓'] }
    )
    // merged into the holder row, not dropped
    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', tools: ['Terminal("one") ✓', 'Terminal("two") ✓'] }
    ])
  })
})
