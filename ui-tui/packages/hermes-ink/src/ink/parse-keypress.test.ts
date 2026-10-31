import { describe, expect, it } from 'vitest'

import { BURST_END_QUIET_MS, INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'
import { PASTE_END, PASTE_START } from './termio/csi.js'

describe('legacy modified return parsing', () => {
  it.each(['\r', '\n'])('parses ESC+%j as one Alt+Enter keypress', lineEnding => {
    const sequence = `\x1b${lineEnding}`
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, sequence)

    expect(keys).toEqual([expect.objectContaining({ name: 'return', ctrl: false, meta: true, shift: false, sequence })])
  })
})

describe('legacy function-key parsing', () => {
  it.each([
    ['f1', '\x1bOP'],
    ['f2', '\x1bOQ'],
    ['f3', '\x1bOR'],
    ['f4', '\x1bOS'],
    ['f5', '\x1b[15~'],
    ['f6', '\x1b[17~'],
    ['f7', '\x1b[18~'],
    ['f8', '\x1b[19~'],
    ['f9', '\x1b[20~'],
    ['f10', '\x1b[21~'],
    ['f11', '\x1b[23~'],
    ['f12', '\x1b[24~']
  ])('parses %s from its terminal sequence', (name, sequence) => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, sequence)

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ name, fn: true, ctrl: false, meta: false, shift: false })
  })
})

describe('parseMultipleKeypresses bracketed paste recovery', () => {
  it('emits empty bracketed pastes when the terminal sends both markers', () => {
    const [keys, state] = parseMultipleKeypresses(INITIAL_STATE, PASTE_START + PASTE_END)

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ isPasted: true, raw: '' })
    expect(state.mode).toBe('NORMAL')
  })

  it('flushes unterminated paste content back to normal input mode', () => {
    const [pendingKeys, pendingState] = parseMultipleKeypresses(INITIAL_STATE, PASTE_START + 'hello')

    expect(pendingKeys).toEqual([])
    expect(pendingState.mode).toBe('IN_PASTE')

    const [keys, state] = parseMultipleKeypresses(pendingState, null)

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ isPasted: true, raw: 'hello' })
    expect(state.mode).toBe('NORMAL')
    expect(state.pasteBuffer).toBe('')
  })

  it('resets an empty unterminated paste start instead of staying stuck', () => {
    const [pendingKeys, pendingState] = parseMultipleKeypresses(INITIAL_STATE, PASTE_START)

    expect(pendingKeys).toEqual([])
    expect(pendingState.mode).toBe('IN_PASTE')

    const [keys, state] = parseMultipleKeypresses(pendingState, null)

    expect(keys).toEqual([])
    expect(state.mode).toBe('NORMAL')
    expect(state.pasteBuffer).toBe('')
  })
})

describe('parseMultipleKeypresses text control splitting', () => {
  it('keeps an IME backspace plus composed character in the same read', () => {
    const [keys, state] = parseMultipleKeypresses(INITIAL_STATE, '\x7fô')

    expect(keys).toEqual([
      expect.objectContaining({ name: 'backspace', raw: '\x7f' }),
      expect.objectContaining({ name: '', raw: 'ô' })
    ])
    expect(state.mode).toBe('NORMAL')
  })

  it('keeps trailing IME text after a backspace in the same read', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x7fôi')

    expect(keys).toEqual([
      expect.objectContaining({ name: 'backspace', raw: '\x7f' }),
      expect.objectContaining({ name: '', raw: 'ôi' })
    ])
  })

  it('splits embedded backspace control bytes without splitting surrounding text', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'ab\bç')

    expect(keys).toEqual([
      expect.objectContaining({ name: '', raw: 'ab' }),
      expect.objectContaining({ name: 'backspace', raw: '\b' }),
      expect.objectContaining({ name: '', raw: 'ç' })
    ])
  })

  it('peels off a non-backspace control byte fused with text instead of dropping the whole chunk', () => {
    // An IME can fuse a control byte other than \x7f/\b with the recomposed
    // text (here U+0001). The original PR only split on \x7f/\b, so a chunk
    // like "a\x01b" fell through every parseKeypress branch, returned
    // name:"" with a non-printable sequence, and the composer discarded the
    // entire chunk — eating the printable letters 'a' and 'b' too. Every
    // control byte must be peeled off so the surrounding text survives.
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'a\x01b')

    // The leading and trailing printable letters must each survive as their
    // own keypress (the control byte in between parses to ctrl+a). The bug was
    // the WHOLE "a\x01b" chunk collapsing into one undeliverable key.
    expect(keys).toHaveLength(3)
    expect(keys[0]).toMatchObject({ name: 'a', raw: 'a' })
    expect(keys[1]).toMatchObject({ raw: '\x01' })
    expect(keys[2]).toMatchObject({ name: 'b', raw: 'b' })
  })

  it('keeps printable letters around a fused ESC control byte', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'vương\x1b')

    // The trailing printable run must still be delivered as its own key.
    expect(keys.some(k => 'raw' in k && k.raw === 'vương')).toBe(true)
  })

  it('does NOT split embedded CR/LF (preserves paste/return handling)', () => {
    // CR/LF inside a text token come from non-bracketed paste; splitting them
    // into `return` keys would prematurely submit the composer. They must stay
    // inside the single text token.
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'a\rb')

    expect(keys).toEqual([expect.objectContaining({ raw: 'a\rb' })])
  })

  it('submits a single-line read ending in CR once input stays quiet (scripted text + Enter in one write)', () => {
    // `tmux send-keys '/help' Enter` and a PTY write of `/help\r` arrive as one read.
    const [keys, held] = parseMultipleKeypresses(INITIAL_STATE, '/help\r', 1_000)

    expect(keys).toEqual([expect.objectContaining({ raw: '/help' })])
    expect(held.pendingReturn).toBe(true)

    const [flushed, after] = parseMultipleKeypresses(held, null, 1_000 + BURST_END_QUIET_MS)

    expect(flushed).toEqual([expect.objectContaining({ name: 'return' })])
    expect(after.pendingReturn).toBe(false)
  })

  it('submits a held CR when an Enter read follows inside the window (scripted text + Enter, then Enter)', () => {
    const [keys, held] = parseMultipleKeypresses(INITIAL_STATE, 'alpha\r', 1_000)
    const [next, after] = parseMultipleKeypresses(held, '\r', 1_060)

    expect([...keys, ...next]).toEqual([
      expect.objectContaining({ raw: 'alpha' }),
      expect.objectContaining({ name: 'return' }),
      expect.objectContaining({ name: 'return' })
    ])
    expect(after.pendingReturn).toBe(false)
  })

  it('keeps holding a CR across terminal reports, so a split paste stays content', () => {
    // A focus report, a DECRPM reply, an SGR mouse event and a horizontal
    // wheel event land between the two reads of a non-bracketed paste split
    // right after a CR.
    let [keys, state] = parseMultipleKeypresses(INITIAL_STATE, 'lineA\r', 1_000)

    for (const [i, report] of ['\x1b[I', '\x1b[?1003;2$y', '\x1b[<35;10;5M', '\x1b[<66;10;5M'].entries()) {
      const [parsed, next] = parseMultipleKeypresses(state, report, 1_010 + i * 10)

      keys = [...keys, ...parsed]
      state = next
      expect(state.pendingReturn).toBe(true)
    }

    const [tail, rest] = parseMultipleKeypresses(state, 'lineB', 1_100)
    const [flushed] = parseMultipleKeypresses(rest, null, 1_100 + BURST_END_QUIET_MS)
    const all = [...keys, ...tail, ...flushed]

    expect(all.some(key => key.kind === 'key' && key.name === 'return')).toBe(false)
    expect(all.at(-1)).toEqual(expect.objectContaining({ raw: '\rlineB' }))
  })

  it('submits a held CR and an Enter read in one chunk with a terminal report, in either order', () => {
    const [, held] = parseMultipleKeypresses(INITIAL_STATE, 'alpha\r', 1_000)
    const [keys, after] = parseMultipleKeypresses(held, '\x1b[I\r', 1_010)

    expect(keys.map(key => (key.kind === 'key' ? key.name || key.sequence : key.kind))).toEqual([
      '\x1b[I',
      'return',
      'return'
    ])
    expect(after.pendingReturn).toBe(false)

    const [reversed, afterReversed] = parseMultipleKeypresses(held, '\r\x1b[I', 1_010)

    expect(reversed.map(key => (key.kind === 'key' ? key.name || key.sequence : key.kind))).toEqual([
      'return',
      'return',
      '\x1b[I'
    ])
    expect(afterReversed.pendingReturn).toBe(false)
  })

  it('submits a held CR when an LF Return shares a read with a terminal report, in either order', () => {
    // LF is a Return too (Ctrl+J; it submits on some POSIX setups), so it
    // releases the held CR like Enter and keeps its own sequence.
    const [, held] = parseMultipleKeypresses(INITIAL_STATE, 'alpha\r', 1_000)

    for (const [input, expected] of [
      ['\x1b[I\n', ['\x1b[I', 'return', 'return']],
      ['\x1b[?1003;2$y\n', ['response', 'return', 'return']],
      ['\n\x1b[I', ['return', 'return', '\x1b[I']]
    ] as const) {
      const [keys, after] = parseMultipleKeypresses(held, input, 1_010)
      const returns = keys.filter(key => key.kind === 'key' && key.name === 'return')

      expect(keys.map(key => (key.kind === 'key' ? key.name || key.sequence : key.kind))).toEqual(expected)
      expect(returns.map(key => key.sequence)).toEqual(['\r', '\n'])
      expect(after.pendingReturn).toBe(false)
    }
  })

  it('keeps a held CR as paste content when its read carries more text after a newline and a report', () => {
    // Split unbracketed paste with a blank line at the boundary and a report
    // inside the next read: the newline token is no Enter, the read is a paste.
    const [, held] = parseMultipleKeypresses(INITIAL_STATE, 'lineA\r', 1_000)

    for (const [input, report] of [
      ['\n\x1b[IlineB', '\x1b[I'],
      ['\r\x1b[IlineB', '\x1b[I'],
      ['\n\x1b[?1003;2$ylineB', 'response'],
      ['\r\n\x1b[<35;10;5MlineB', 'mouse']
    ] as const) {
      const [keys, after] = parseMultipleKeypresses(held, input, 1_010)
      const [flushed] = parseMultipleKeypresses(after, null, 1_010 + BURST_END_QUIET_MS)
      const all = [...keys, ...flushed]

      expect(all.some(key => key.kind === 'key' && key.name === 'return')).toBe(false)
      expect(all.map(key => (key.kind === 'key' ? key.name || key.raw : key.kind))).toEqual([
        '\r' + input.slice(0, input.indexOf('\x1b')),
        report,
        'lineB'
      ])
    }
  })

  it('keeps a held CR as paste content when the next read is several line breaks', () => {
    // Only a read whose text is one CR or LF is an Enter; '\r\n', '\r\r' or
    // '\n\n' after a held CR stays one pasted text run, as before the hold
    // learned about Enter.
    const [, held] = parseMultipleKeypresses(INITIAL_STATE, 'lineA\r', 1_000)

    for (const input of ['\r\n', '\r\r', '\n\n']) {
      const [keys, after] = parseMultipleKeypresses(held, input, 1_010)

      expect(keys).toEqual([expect.objectContaining({ raw: '\r' + input })])
      expect(after.pendingReturn).toBe(false)
    }
  })

  it('ends the hold one window after the CR, however long mouse motion keeps streaming', () => {
    let [, state] = parseMultipleKeypresses(INITIAL_STATE, 'alpha\r', 1_000)
    const released: number[] = []

    for (let now = 1_010; now <= 1_300; now += 30) {
      const [keys, next] = parseMultipleKeypresses(state, '\x1b[<35;10;5M', now)

      if (keys.some(key => key.kind === 'key' && key.name === 'return')) {
        released.push(now)
      }

      state = next
    }

    expect(released).toEqual([1_250])
    expect(state.pendingReturn).toBe(false)
  })

  it('releases a held CR before a key escape that follows a terminal report', () => {
    const [, held] = parseMultipleKeypresses(INITIAL_STATE, 'alpha\r', 1_000)
    const [keys, after] = parseMultipleKeypresses(held, '\x1b[I\x1b[A', 1_010)

    expect(keys.map(key => (key.kind === 'key' ? key.name || key.sequence : key.kind))).toEqual([
      '\x1b[I',
      'return',
      'up'
    ])
    expect(after.pendingReturn).toBe(false)
  })

  it('keeps every CR of a non-bracketed paste split across reads as content', () => {
    // Tail read: 'line one\rline ' then 'two\r' inside the quiet window.
    const [first, mid] = parseMultipleKeypresses(INITIAL_STATE, 'line one\rline ', 1_000)
    const [second, end] = parseMultipleKeypresses(mid, 'two\r', 1_010)
    const [flushed] = parseMultipleKeypresses(end, null, 1_010 + BURST_END_QUIET_MS)

    expect([...first, ...second, ...flushed]).toEqual([
      expect.objectContaining({ raw: 'line one\rline ' }),
      expect.objectContaining({ raw: 'two\r' })
    ])

    // Head read: 'line one\r' then 'line two' inside the quiet window.
    const [a, held] = parseMultipleKeypresses(INITIAL_STATE, 'line one\r', 2_000)
    const [b, rest] = parseMultipleKeypresses(held, 'line two', 2_010)
    const [c] = parseMultipleKeypresses(rest, null, 2_010 + BURST_END_QUIET_MS)

    expect([...a, ...b, ...c]).toEqual([
      expect.objectContaining({ raw: 'line one' }),
      expect.objectContaining({ raw: '\rline two' })
    ])
  })

  it('keeps the trailing CR of a paste whose reads keep arriving past the quiet window', () => {
    // 'one\r', 'two', ' continued', ' again', ' end\r' 100 ms apart: the last break is older than
    // BURST_END_QUIET_MS, but input never went quiet, so the trailing CR is still paste content.
    let state = INITIAL_STATE
    const keys = []

    for (const [i, chunk] of ['one\r', 'two', ' continued', ' again', ' end\r'].entries()) {
      const [parsed, next] = parseMultipleKeypresses(state, chunk, 1_000 + i * 100)

      keys.push(...parsed)
      state = next
    }

    const [flushed] = parseMultipleKeypresses(state, null, 1_400 + BURST_END_QUIET_MS)

    keys.push(...flushed)
    expect(keys.some(key => 'name' in key && key.name === 'return')).toBe(false)
    expect(keys.at(-1)).toEqual(expect.objectContaining({ raw: ' end\r' }))
  })

  it('keeps the trailing CR of a multi-line non-bracketed paste inside the text token', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'line one\rline two\r')

    expect(keys).toEqual([expect.objectContaining({ raw: 'line one\rline two\r' })])
  })
})

describe('parseMultipleKeypresses CSI u (Kitty keyboard protocol)', () => {
  it('parses Cmd+Enter (CSI 13;9u) as return with super=true', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[13;9u')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({
      kind: 'key',
      name: 'return',
      shift: false,
      ctrl: false,
      meta: false,
      super: true
    })
  })

  it('parses plain Enter (CSI 13u) as return with no modifiers', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[13u')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({
      kind: 'key',
      name: 'return',
      shift: false,
      ctrl: false,
      meta: false,
      super: false
    })
  })
})

describe('mouse wheel modifier decoding', () => {
  // SGR mouse format: ESC [ < button ; col ; row M
  // Wheel up = 64 (0x40), wheel down = 65 (0x41).
  // Modifier bits: shift = 0x04, meta = 0x08, ctrl = 0x10.
  const sgrWheel = (button: number) => `\x1b[<${button};10;10M`

  it('plain wheel up has no modifiers', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, sgrWheel(0x40))

    expect(key).toMatchObject({ name: 'wheelup', ctrl: false, meta: false, shift: false })
  })

  it('decodes meta (Alt/Option) on wheel up', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, sgrWheel(0x40 | 0x08))

    expect(key).toMatchObject({ name: 'wheelup', ctrl: false, meta: true, shift: false })
  })

  it('decodes ctrl on wheel events', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, sgrWheel(0x40 | 0x10))

    expect(key).toMatchObject({ name: 'wheelup', ctrl: true, meta: false, shift: false })
  })

  it('decodes shift on wheel events', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, sgrWheel(0x41 | 0x04))

    expect(key).toMatchObject({ name: 'wheeldown', ctrl: false, meta: false, shift: true })
  })

  it('decodes combined modifiers', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, sgrWheel(0x40 | 0x08 | 0x10))

    expect(key).toMatchObject({ name: 'wheelup', ctrl: true, meta: true, shift: false })
  })

  it('decodes meta on legacy X10 wheel encoding', () => {
    // X10: ESC [ M Cb Cx Cy where each byte is value+32.
    const x10 = `\x1b[M${String.fromCharCode(0x40 + 0x08 + 32)}${String.fromCharCode(10 + 32)}${String.fromCharCode(10 + 32)}`
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, x10)

    expect(key).toMatchObject({ name: 'wheelup', meta: true })
  })
})

describe('flush-boundary SGR mouse reassembly', () => {
  it('reassembles a report split by a mid-sequence watchdog flush into one mouse event', () => {
    // chunk 1: heavy render stalls the loop, only the prefix is read
    let [keys, state] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[<0;35;')
    expect(keys).toEqual([])

    // App's 50ms watchdog flushes (input=null) — must NOT emit the partial
    ;[keys, state] = parseMultipleKeypresses(state, null)
    expect(keys).toEqual([])

    // continuation arrives; the whole report reassembles, nothing leaks
    ;[keys, state] = parseMultipleKeypresses(state, '46M')
    expect(keys).toEqual([expect.objectContaining({ kind: 'mouse', button: 0, col: 35, row: 46, action: 'press' })])
  })

  it('re-synthesizes an orphaned X10 wheel tail (legacy mouse) into a scroll key', () => {
    // X10 wheel-up = ESC[M + (0x40+32) + col + row. If the ESC was flushed as a
    // lone Escape and the `[M…` payload arrives as text, resynthesize it.
    const tail = '[M' + String.fromCharCode(0x60) + '!!'
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, tail)

    expect(key).toMatchObject({ name: 'wheelup' })
  })
})

describe('cursor position report parsing', () => {
  it('parses DECXCPR cursor position report (CSI ? row;col R)', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[?22;1R')

    expect(key).toMatchObject({
      kind: 'response',
      response: { type: 'cursorPosition', row: 22, col: 1 }
    })
  })

  it('parses standard DSR cursor position report (CSI row;col R) when row > 1', () => {
    // Terminals that don't support DECXCPR may respond to CSI ? 6 n with
    // the plain DSR form (no ?). These must be recognized as responses,
    // not inserted as literal text.
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[22;1R')

    expect(key).toMatchObject({
      kind: 'response',
      response: { type: 'cursorPosition', row: 22, col: 1 }
    })
  })

  it('does NOT treat CSI 1;2 R as a cursor position report (Shift+F3 ambiguity)', () => {
    // CSI 1;2 R is Shift+F3 in xterm. Without the ? marker, row 1 is
    // ambiguous with F3 modifiers — must fall through to parseKeypress,
    // not be silently dropped as a terminal response.
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[1;2R')

    expect(key.kind).not.toBe('response')
  })

  it('does NOT treat CSI 0;col R as a cursor position report (invalid row-zero DSR)', () => {
    // Terminal coordinates are 1-indexed, so a plain DSR report with row 0 is
    // invalid. Without the ? marker it must remain unclassified rather than be
    // reported as a cursor position (guard is row <= 1, not row === 1).
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[0;5R')

    expect(key.kind).not.toBe('response')
  })

  it('treats DECXCPR at row 1 as a cursor position report (? disambiguates from F3)', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[?1;2R')

    expect(key).toMatchObject({
      kind: 'response',
      response: { type: 'cursorPosition', row: 1, col: 2 }
    })
  })
})
