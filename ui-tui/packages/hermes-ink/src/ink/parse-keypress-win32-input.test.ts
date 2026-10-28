import { describe, expect, it } from 'vitest'

import { InputEvent } from './events/input-event.js'
import { INITIAL_STATE, type ParsedInput, parseMultipleKeypresses } from './parse-keypress.js'
import { PASTE_END, PASTE_START } from './termio/csi.js'

// ConPTY win32-input-mode (microsoft/terminal spec #4999): every KEY_EVENT_RECORD
// is sent as `CSI Vk;Sc;Uc;Kd;Cs;Rc _`, one record per key-down AND per key-up.
// A pane console left in that mode fed these records to the TUI, which had no
// decoder, so the composer filled with `[79;24;79;1;48;1_` text and Ctrl+C never
// fired. Spec defaults: omitted Vk/Sc/Uc/Kd/Cs = 0, omitted Rc = 1.
const ESC = '\x1b'

const rec = (vk: number, sc: number, uc: number, kd: 0 | 1, cs: number, rc = 1): string =>
  `${ESC}[${vk};${sc};${uc};${kd};${cs};${rc}_`

// dwControlKeyState bits (Win32 KEY_EVENT_RECORD)
const RIGHT_ALT = 0x0001
const LEFT_ALT = 0x0002
const LEFT_CTRL = 0x0008
const SHIFT = 0x0010
const NUMLOCK = 0x0020
const ENHANCED = 0x0100

const PRINTABLE = /^[ -~\u00a0-\uffff]+$/

/** Mirror the composer's insert gate (ui-tui/src/components/textInput.tsx). */
const composerWouldInsert = (event: InputEvent): boolean =>
  !event.isControlChord && (event.keypress.isPasted || event.input.length > 0) && PRINTABLE.test(event.input)

/** Feed chunks one stdin read at a time (null = watchdog flush), carrying parser state. */
function parseAll(feeds: Array<string | null>): ParsedInput[] {
  let state = { ...INITIAL_STATE }
  const out: ParsedInput[] = []

  for (const feed of feeds) {
    const [keys, next] = parseMultipleKeypresses(state, feed)
    out.push(...keys)
    state = next
  }

  return out
}

/** What the composer would end up inserting for the parsed keys (special keys => ''). */
function composerText(keys: ParsedInput[]): string[] {
  return keys.map(k => {
    const event = new InputEvent(k as never)

    return composerWouldInsert(event) ? event.input : ''
  })
}

// The 14 records captured in the live composer (typed `Once `, Backspace, Ctrl+C, NumLock on).
const captured = [
  rec(79, 24, 79, 1, SHIFT | NUMLOCK),
  rec(79, 24, 79, 0, SHIFT | NUMLOCK),
  rec(78, 49, 110, 1, NUMLOCK),
  rec(78, 49, 110, 0, NUMLOCK),
  rec(67, 46, 99, 1, NUMLOCK),
  rec(69, 18, 101, 1, NUMLOCK),
  rec(67, 46, 99, 0, NUMLOCK),
  rec(69, 18, 101, 0, NUMLOCK),
  rec(32, 57, 32, 1, NUMLOCK),
  rec(32, 57, 32, 0, NUMLOCK),
  rec(8, 14, 8, 1, NUMLOCK),
  rec(8, 14, 8, 0, NUMLOCK),
  rec(67, 46, 3, 1, LEFT_CTRL | NUMLOCK),
  rec(67, 46, 3, 0, LEFT_CTRL | NUMLOCK)
]

const capturedNames = ['o', 'n', 'c', 'e', 'space', 'backspace', 'c']
const capturedText = ['O', 'n', 'c', 'e', ' ', '', '']

describe('win32-input-mode records: captured incident input', () => {
  it('decodes the records one per read as the typed keys, dropping key-ups', () => {
    const keys = parseAll(captured)

    expect(keys.map(k => (k.kind === 'key' ? k.name : k.kind))).toEqual(capturedNames)
    expect(composerText(keys)).toEqual(capturedText)
    expect(keys[0]).toMatchObject({ name: 'o', shift: true, ctrl: false, meta: false })
    expect(keys[6]).toMatchObject({ name: 'c', ctrl: true })
    expect(new InputEvent(keys[6] as never).isControlChord).toBe(true)
  })

  it('decodes the same records coalesced into one read', () => {
    const keys = parseAll([captured.join('')])

    expect(keys.map(k => (k.kind === 'key' ? k.name : k.kind))).toEqual(capturedNames)
    expect(composerText(keys)).toEqual(capturedText)
  })

  it('never lets record text reach the composer', () => {
    for (const key of parseAll([captured.join('')])) {
      expect(new InputEvent(key as never).input).not.toContain('[')
    }
  })
})

describe('win32-input-mode records: split across reads', () => {
  const shiftO = captured[0]!

  it.each([
    ['after ESC', [ESC, shiftO.slice(1)], 1],
    ['mid-parameters', [shiftO.slice(0, 7), shiftO.slice(7)], 1],
    ['before the final _', [shiftO.slice(0, -1), '_'], 1],
    ['byte by byte', [...shiftO], 1],
    ['across a watchdog flush mid-record', [shiftO.slice(0, 9), null, shiftO.slice(9)], 1],
    ['inside the second of two records', [shiftO + captured[2]!.slice(0, 5), captured[2]!.slice(5)], 2]
  ] as Array<[string, Array<string | null>, number]>)('reassembles a record split %s', (_label, feeds, count) => {
    const keys = parseAll(feeds)

    expect(keys).toHaveLength(count)
    expect(keys[0]).toMatchObject({ kind: 'key', name: 'o', shift: true })
    expect(composerText(keys)[0]).toBe('O')
  })
})

describe('win32-input-mode records: defaults, validation and repeat count', () => {
  it('drops key-up records', () => {
    expect(parseAll([rec(65, 30, 97, 0, 0)])).toEqual([])
  })

  it.each([
    ['trailing Cs;Rc omitted', `${ESC}[65;30;97;1_`],
    ['Vk and Sc empty', `${ESC}[;;97;1;;_`],
    ['every parameter but Uc and Kd empty', `${ESC}[;;97;1_`]
  ])('applies spec defaults when parameters are omitted (%s)', (_label, sequence) => {
    const keys = parseAll([sequence])

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ name: 'a', sequence: 'a' })
  })

  it('treats an omitted Kd as key-up (spec default 0)', () => {
    expect(parseAll([`${ESC}[;;97_`])).toEqual([])
    expect(parseAll([`${ESC}[_`])).toEqual([])
  })

  it('honours the repeat count', () => {
    const keys = parseAll([rec(65, 30, 97, 1, 0, 3)])

    expect(keys.map(k => (k.kind === 'key' ? k.sequence : k.kind))).toEqual(['a', 'a', 'a'])
  })

  it('bounds a pathological repeat count instead of expanding it', () => {
    expect(parseAll([rec(65, 30, 97, 1, 0, 1_000_000)])).toHaveLength(32)
    expect(parseAll([rec(65, 30, 97, 1, 0, 0)])).toHaveLength(1)
  })

  it('drops records whose character is not a UTF-16 code unit', () => {
    expect(parseAll([rec(0, 0, 70_000, 1, 0)])).toEqual([])
  })

  it('drops modifier-only and unknown keys with no character', () => {
    expect(parseAll([rec(16, 42, 0, 1, SHIFT), rec(17, 29, 0, 1, LEFT_CTRL), rec(255, 0, 0, 1, 0)])).toEqual([])
  })
})

describe('win32-input-mode records: special keys and modifiers', () => {
  it.each([
    ['left', 37, 0],
    ['up', 38, 0],
    ['right', 39, 0],
    ['down', 40, 0],
    ['home', 36, 0],
    ['end', 35, 0],
    ['insert', 45, 0],
    ['delete', 46, 0],
    ['pageup', 33, 0],
    ['pagedown', 34, 0],
    ['f1', 112, 0],
    ['f5', 116, 0],
    ['f12', 123, 0],
    ['return', 13, 13],
    ['escape', 27, 27],
    ['tab', 9, 9],
    ['backspace', 8, 8]
  ])('maps %s (Vk %i, Uc %i) to the legacy key name', (name, vk, uc) => {
    const keys = parseAll([rec(vk, 0, uc, 1, ENHANCED | NUMLOCK)])

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ kind: 'key', name, ctrl: false, meta: false })
    expect(composerText(keys)).toEqual([''])
  })

  it('carries Ctrl/Shift/Alt onto navigation keys', () => {
    expect(parseAll([rec(37, 75, 0, 1, LEFT_CTRL | ENHANCED)])[0]).toMatchObject({ name: 'left', ctrl: true })
    expect(parseAll([rec(37, 75, 0, 1, SHIFT | ENHANCED)])[0]).toMatchObject({ name: 'left', shift: true })
    expect(parseAll([rec(37, 75, 0, 1, LEFT_ALT | ENHANCED)])[0]).toMatchObject({ name: 'left', meta: true })
    expect(parseAll([rec(46, 83, 0, 1, LEFT_CTRL | ENHANCED)])[0]).toMatchObject({ name: 'delete', ctrl: true })
  })

  it('maps Shift+Tab to the backtab key', () => {
    expect(parseAll([rec(9, 15, 9, 1, SHIFT)])[0]).toMatchObject({ name: 'tab', shift: true })
  })

  it('decodes Ctrl chords from their control character', () => {
    const [ctrlA] = parseAll([rec(65, 30, 1, 1, LEFT_CTRL)])

    expect(ctrlA).toMatchObject({ name: 'a', ctrl: true })
    expect(new InputEvent(ctrlA as never).isControlChord).toBe(true)
  })

  it('decodes Alt+letter and Alt+Enter the way the legacy ESC prefix does', () => {
    const [altX] = parseAll([rec(88, 45, 120, 1, LEFT_ALT)])
    const [altEnter] = parseAll([rec(13, 28, 13, 1, LEFT_ALT)])

    expect(altX).toMatchObject({ meta: true, sequence: `${ESC}x` })
    expect(altEnter).toMatchObject({ name: 'return', meta: true })
  })

  it('treats AltGr characters as plain text', () => {
    const keys = parseAll([rec(50, 3, 64, 1, LEFT_CTRL | RIGHT_ALT)])

    expect(keys[0]).toMatchObject({ sequence: '@', ctrl: false, meta: false })
    expect(composerText(keys)).toEqual(['@'])
  })

  it('keeps Shift on uppercase letters', () => {
    const keys = parseAll([rec(79, 24, 79, 1, SHIFT)])

    expect(keys[0]).toMatchObject({ name: 'o', shift: true })
    expect(composerText(keys)).toEqual(['O'])
  })
})

describe('win32-input-mode records: Unicode', () => {
  it('inserts a BMP character', () => {
    expect(composerText(parseAll([rec(0, 0, 233, 1, 0)]))).toEqual(['é'])
  })

  it('joins a surrogate pair sent as two records, even across reads', () => {
    const high = rec(0, 0, 0xd83d, 1, 0)
    const low = rec(0, 0, 0xde00, 1, 0)

    expect(composerText(parseAll([high + low]))).toEqual(['😀'])
    expect(composerText(parseAll([high, low]))).toEqual(['😀'])
    // The high half's key-up arrives before the low half's key-down.
    expect(composerText(parseAll([high, rec(0, 0, 0xd83d, 0, 0), low]))).toEqual(['😀'])
  })

  it('drops an unpaired surrogate half instead of inserting it', () => {
    expect(parseAll([rec(0, 0, 0xde00, 1, 0)])).toEqual([])
    expect(composerText(parseAll([rec(0, 0, 0xd83d, 1, 0), rec(65, 30, 97, 1, 0)]))).toEqual(['a'])
  })
})

describe('win32-input-mode records: legacy input is untouched', () => {
  it.each([
    ['a', 'a'],
    ['\x03', 'c'],
    [`${ESC}[D`, 'left'],
    [`${ESC}[A`, 'up'],
    ['\r', 'return'],
    ['\x7f', 'backspace'],
    [`${ESC}[15~`, 'f5'],
    [`${ESC}[13;2u`, 'return']
  ])('parses %j as %s', (sequence, name) => {
    expect(parseAll([sequence])[0]).toMatchObject({ kind: 'key', name })
  })

  it('keeps record text literal inside a bracketed paste', () => {
    const keys = parseAll([PASTE_START + captured[0] + PASTE_END])

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ isPasted: true, sequence: captured[0] })
  })

  it('still emits a lone Escape on flush', () => {
    expect(parseAll([ESC, null])[0]).toMatchObject({ name: 'escape' })
  })
})
