import { describe, expect, it } from 'vitest'

import { createTokenizer } from './tokenize.js'

describe('tokenizer escape-sequence boundaries', () => {
  it('reassembles a CSI mouse sequence split across two feeds', () => {
    const t = createTokenizer({ x10Mouse: true })

    expect(t.feed('\x1b[<0;35;')).toEqual([])
    expect(t.feed('46M')).toEqual([{ type: 'sequence', value: '\x1b[<0;35;46M' }])
    expect(t.buffer()).toBe('')
  })
})

describe('tokenizer state-aware flush', () => {
  it('does not emit an incomplete CSI on flush — it keeps it for reassembly', () => {
    const t = createTokenizer({ x10Mouse: true })

    // A render stall lets App's watchdog flush mid-sequence. The buffered CSI
    // prefix must NOT be emitted (that is the `46M…` leak); it stays buffered.
    expect(t.feed('\x1b[<0;35;')).toEqual([])
    expect(t.flush()).toEqual([])
    expect(t.buffer()).toBe('\x1b[<0;35;')

    // The continuation arrives on the next feed and the whole report
    // reassembles into a single clean sequence token — nothing leaked.
    expect(t.feed('46M')).toEqual([{ type: 'sequence', value: '\x1b[<0;35;46M' }])
    expect(t.buffer()).toBe('')
  })

  it('drops a partial control sequence that survives a second flush (truncation)', () => {
    const t = createTokenizer({ x10Mouse: true })

    expect(t.feed('\x1b[<0;35;')).toEqual([])
    expect(t.flush()).toEqual([]) // first flush keeps the buffer
    expect(t.buffer()).toBe('\x1b[<0;35;')

    // Continuation never arrived: the next flush sees the same buffer and
    // drops it so it can't fuse with the next keypress's bytes.
    expect(t.flush()).toEqual([])
    expect(t.buffer()).toBe('')
  })

  it('still emits a bare ESC on flush so the Escape key works', () => {
    const t = createTokenizer({ x10Mouse: true })

    expect(t.feed('\x1b')).toEqual([])
    expect(t.flush()).toEqual([{ type: 'sequence', value: '\x1b' }])
    expect(t.buffer()).toBe('')
  })

  it('reassembles even when a flush fires between every byte of the report', () => {
    const t = createTokenizer({ x10Mouse: true })

    // Pathological stall: a flush between each chunk. As long as the
    // continuation eventually arrives, no fragment is ever emitted as input.
    for (const chunk of ['\x1b[', '<', '0;', '35;', '46']) {
      expect(t.feed(chunk)).toEqual([])
      expect(t.flush()).toEqual([])
    }

    expect(t.feed('M')).toEqual([{ type: 'sequence', value: '\x1b[<0;35;46M' }])
    expect(t.buffer()).toBe('')
  })
})
