import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'
import { PASTE_END, PASTE_START } from './termio/csi.js'

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

  it('plain wheel down has no modifiers', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, sgrWheel(0x41))

    expect(key).toMatchObject({ name: 'wheeldown', ctrl: false, meta: false, shift: false })
  })

  it('decodes meta (Alt/Option) on wheel up', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, sgrWheel(0x40 | 0x08))

    expect(key).toMatchObject({ name: 'wheelup', ctrl: false, meta: true, shift: false })
  })

  it('decodes meta (Alt/Option) on wheel down', () => {
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, sgrWheel(0x41 | 0x08))

    expect(key).toMatchObject({ name: 'wheeldown', ctrl: false, meta: true, shift: false })
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

  it('drops a truncated mouse prefix after a second flush instead of leaking it', () => {
    let [keys, state] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[<0;35;')

    ;[keys, state] = parseMultipleKeypresses(state, null) // first flush keeps it
    ;[keys, state] = parseMultipleKeypresses(state, null) // second flush drops it

    expect(keys).toEqual([])
    expect(state.incomplete).toBe('')
  })

  it('re-synthesizes an orphaned X10 wheel tail (legacy mouse) into a scroll key', () => {
    // X10 wheel-up = ESC[M + (0x40+32) + col + row. If the ESC was flushed as a
    // lone Escape and the `[M…` payload arrives as text, resynthesize it.
    const tail = '[M' + String.fromCharCode(0x60) + '!!'
    const [[key]] = parseMultipleKeypresses(INITIAL_STATE, tail)

    expect(key).toMatchObject({ name: 'wheelup' })
  })
})
