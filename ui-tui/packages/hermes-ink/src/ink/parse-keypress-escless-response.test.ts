import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'

// A terminal reply can reach the TUI with its leading ESC stripped by the
// transport — observed on the Proxmox web console path, where the reply body
// `ESC[?1000;2$y` is delivered as the bare printable text `[?1000;2$y`. The
// tokenizer classifies it as text and parseTextKeypresses then types it into
// the composer, so the user sees "random" escape text mid-sentence.
//
// These cases pin the parser's handling of that shape: a DEC-private reply
// body must be recognized as a terminal response (and thus dropped when no
// query is pending), never inserted as text.

type AnyKey = { kind?: string; sequence?: string }

const feed = (input: string) => parseMultipleKeypresses(INITIAL_STATE, input)[0]

// Mirrors the composer's PRINTABLE gate: what would actually be inserted.
const insertedText = (keys: unknown[]) =>
  (keys as AnyKey[])
    .filter(k => k.kind !== 'response' && k.kind !== 'mouse')
    .map(k => k.sequence ?? '')
    .filter(seq => /^[ -~\u00a0-\uffff]+$/.test(seq))
    .join('')

describe('ESC-less terminal responses', () => {
  it.each([
    ['DECRPM for mouse mode 1000', '[?1000;2$y', { type: 'decrpm', mode: 1000, status: 2 }],
    ['DECRPM for mode 2026', '[?2026;2$y', { type: 'decrpm', mode: 2026, status: 2 }],
    ['DA1 body', '[?1000;2c', { type: 'da1', params: [1000, 2] }]
  ])('recognizes %s', (_label, input, expected) => {
    expect(feed(input)).toEqual([expect.objectContaining({ kind: 'response', response: expected })])
  })

  it('does not type the reply body into the composer', () => {
    expect(insertedText(feed('[?1000;2$y'))).toBe('')
  })

  it('lifts a fused reply out of the surrounding typing', () => {
    const keys = feed('ab[?1000;2$ycd')

    expect(keys.map(k => k.kind)).toEqual(['key', 'response', 'key'])
    expect(insertedText(keys)).toBe('abcd')
  })

  it('still recognizes the ESC-prefixed form', () => {
    expect(feed('\x1b[?1000;2$y')).toEqual([
      expect.objectContaining({ kind: 'response', response: { type: 'decrpm', mode: 1000, status: 2 } })
    ])
  })

  it.each([
    ['bracketed word', '[world]'],
    ['bracket digits', '[2;3]'],
    ['question bracket', '[?abc]']
  ])('leaves ordinary %s alone', (_label, input) => {
    const keys = feed(input)

    expect(keys.every(k => k.kind === 'key')).toBe(true)
    expect(insertedText(keys)).toBe(input)
  })
})
