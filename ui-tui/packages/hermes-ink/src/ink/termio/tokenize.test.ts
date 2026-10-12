import { describe, expect, it } from 'vitest'

import { createTokenizer, type Token } from './tokenize.js'

describe('tokenizer escape-sequence boundaries', () => {
  it.each(['\r', '\n'])('keeps ESC+%j together when received in one feed', lineEnding => {
    const t = createTokenizer({ legacyAltEnter: true })
    const sequence = `\x1b${lineEnding}`

    expect(t.feed(sequence)).toEqual([{ type: 'sequence', value: sequence }])
    expect(t.buffer()).toBe('')
  })

  it.each(['\r', '\n'])('reassembles ESC+%j split across two feeds', lineEnding => {
    const t = createTokenizer({ legacyAltEnter: true })
    const sequence = `\x1b${lineEnding}`

    expect(t.feed('\x1b')).toEqual([])
    expect(t.feed(lineEnding)).toEqual([{ type: 'sequence', value: sequence }])
    expect(t.buffer()).toBe('')
  })

  it.each(['\r', '\n'])('keeps Escape distinct when it is flushed before %j', lineEnding => {
    const t = createTokenizer({ legacyAltEnter: true })

    expect(t.feed('\x1b')).toEqual([])
    expect(t.flush()).toEqual([{ type: 'sequence', value: '\x1b' }])
    expect(t.feed(lineEnding)).toEqual([{ type: 'text', value: lineEnding }])
  })
})

describe('tokenizer state-aware flush', () => {
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
})

// Battle-test: prove the leak class is structurally impossible, not just that
// the known cases are patched. We hammer the tokenizer with the worst stalls a
// terminal can produce (split + flush at every byte) and assert the two hard
// invariants: nothing leaks as text, and every complete report reassembles.
describe('tokenizer fuzz: fragments never leak under a flush storm', () => {
  const sgr = (btn: number, col: number, row: number, press: boolean): string =>
    `\x1b[<${btn};${col};${row}${press ? 'M' : 'm'}`

  it('reassembles a report split + flushed at every interior byte', () => {
    const seq = sgr(0, 35, 46, true)

    // Start at 2: an earlier split is the lone-ESC ESCDELAY boundary, which
    // intentionally flushes to the Escape key. Terminals never split a mouse
    // report there — a report is one atomic write — so it's not a real case.
    for (let i = 2; i < seq.length; i++) {
      const t = createTokenizer({ x10Mouse: true })
      const tokens: Token[] = [...t.feed(seq.slice(0, i)), ...t.flush(), ...t.feed(seq.slice(i))]

      expect(tokens).toEqual([{ type: 'sequence', value: seq }])
      expect(t.buffer()).toBe('')
    }
  })

  it('feeds 200 random reports one byte at a time, flushing after every byte', () => {
    // Deterministic PRNG so a failure is reproducible.
    let s = 0x1234567

    const rnd = (n: number): number => {
      s = (s * 1103515245 + 12345) & 0x7fffffff

      return s % n
    }

    const reports = Array.from({ length: 200 }, () => sgr(rnd(120), 1 + rnd(300), 1 + rnd(200), rnd(2) === 0))
    const stream = reports.join('')

    const t = createTokenizer({ x10Mouse: true })
    const seqTokens: string[] = []
    let textLeak = ''

    const drain = (tokens: Token[]): void => {
      for (const tok of tokens) {
        if (tok.type === 'sequence') {
          seqTokens.push(tok.value)
        } else {
          textLeak += tok.value
        }
      }
    }

    for (const ch of stream) {
      drain(t.feed(ch))

      // Flush storm — but not at a lone-ESC boundary (the real watchdog
      // re-arms while bytes are pending; a single flush between feeds never
      // hits the truncation valve).
      if (t.buffer() !== '\x1b') {
        drain(t.flush())
      }
    }

    expect(textLeak).toBe('')
    expect(seqTokens.join('')).toBe(stream)
  })

  it('keeps real keystrokes intact while mouse reports reassemble around them', () => {
    let s = 0x0badf00d

    const rnd = (n: number): number => {
      s = (s * 1103515245 + 12345) & 0x7fffffff

      return s % n
    }

    const typed = 'abc 123 xyz'
    const expectedKeys: string[] = []
    const expectedSeqs: string[] = []
    const parts: string[] = []

    for (let k = 0; k < 120; k++) {
      if (rnd(3) === 0) {
        const ch = typed[rnd(typed.length)]!
        expectedKeys.push(ch)
        parts.push(ch)
      } else {
        const seq = sgr(rnd(64), 1 + rnd(200), 1 + rnd(100), rnd(2) === 0)
        expectedSeqs.push(seq)
        parts.push(seq)
      }
    }

    const stream = parts.join('')
    const t = createTokenizer({ x10Mouse: true })
    const seqTokens: string[] = []
    let text = ''

    const drain = (tokens: Token[]): void => {
      for (const tok of tokens) {
        if (tok.type === 'sequence') {
          seqTokens.push(tok.value)
        } else {
          text += tok.value
        }
      }
    }

    for (const ch of stream) {
      drain(t.feed(ch))

      if (t.buffer() !== '\x1b') {
        drain(t.flush())
      }
    }

    // Every typed character survives, in order; every report reassembles whole.
    expect(text).toBe(expectedKeys.join(''))
    expect(seqTokens).toEqual(expectedSeqs)
  })
})
