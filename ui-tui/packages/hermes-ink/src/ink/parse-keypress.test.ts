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

describe('parseMultipleKeypresses control-byte boundaries', () => {
  it('preserves DEL as a standalone backspace between adjacent IME text chunks', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'ㅎ\x7fㅏ')

    expect(keys).toHaveLength(3)
    expect(keys[0]).toMatchObject({ kind: 'key', raw: 'ㅎ' })
    expect(keys[1]).toMatchObject({ kind: 'key', name: 'backspace', raw: '\x7f' })
    expect(keys[2]).toMatchObject({ kind: 'key', raw: 'ㅏ' })
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
