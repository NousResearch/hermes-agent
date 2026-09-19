import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

// Regression: #115284 — a stray `l` in the TUI composer after a session
// resume / tab switch / window restore.
//
// The dashboard writes the PTY force-redraw byte (Ctrl+L, 0x0c —
// `hermes_cli/pty_session.py` ``TUI_FORCE_REDRAW``) into the TUI's stdin on
// every re-attach (`hermes_cli/web_routers/chat_ws.py`:
// ``session.attach(ws, force_redraw=not _created)``). parse-keypress names a
// control byte after the letter it encodes (0x0c → 'l') and InputEvent hands
// that name to ``input`` so bindings can match ctrl+<letter> on the raw byte —
// so the byte arrived at the composer looking like typed text and landed in the
// input box as a solitary `l`, prefixed to whatever the user typed next.

const PRINTABLE = /^[ -~\u00a0-\uffff]+$/

/** Mirror the composer's insert gate (ui-tui/src/components/textInput.tsx). */
const composerWouldInsert = (event: InputEvent): boolean =>
  !event.isControlByteChord && (event.keypress.isPasted || event.input.length > 0) && PRINTABLE.test(event.input)

function parseOne(bytes: string): InputEvent {
  const [keys] = parseMultipleKeypresses({ ...INITIAL_STATE }, bytes)

  return new InputEvent(keys[0] as never)
}

describe('control bytes are chords, not typed text (#115284)', () => {
  it('does not offer the re-attach redraw byte (Ctrl+L) to inserters', () => {
    const event = parseOne('\x0c')

    expect(event.key.ctrl).toBe(true)
    expect(event.input).toBe('l') // binding name Ink derives from the byte
    expect(composerWouldInsert(event)).toBe(false)
    expect(event.isControlByteChord).toBe(true)
  })

  it('still inserts a typed l', () => {
    const event = parseOne('l')

    expect(event.input).toBe('l')
    expect(event.isControlByteChord).toBe(false)
    expect(composerWouldInsert(event)).toBe(true)
  })

  it('keeps the derived letter on `input` so ctrl chords stay bindable', () => {
    // Clearing `input` instead of flagging it would break the composer's own
    // ctrl+a/e/u/k/w/z/y branches and the global ctrl+c/x/o/t pass-through.
    const chords: Array<[string, string]> = [
      ['\x01', 'a'],
      ['\x05', 'e'],
      ['\x0b', 'k'],
      ['\x15', 'u'],
      ['\x17', 'w'],
      ['\x1a', 'z']
    ]

    for (const [byte, letter] of chords) {
      const event = parseOne(byte)

      expect(event.input, JSON.stringify(byte)).toBe(letter)
      expect(event.key.ctrl, JSON.stringify(byte)).toBe(true)
    }
  })

  it('leaves extended-protocol chords and pastes unflagged', () => {
    // kitty CSI u Ctrl+L is multi-byte, so it is not a bare control byte.
    const kitty = parseOne('\x1b[108;5u')

    expect(kitty.key.ctrl).toBe(true)
    expect(kitty.isControlByteChord).toBe(false)

    // Bracketed paste content is text, never a chord.
    const [keys] = parseMultipleKeypresses({ ...INITIAL_STATE }, '\x1b[200~\x0c\x1b[201~')
    const pasted = new InputEvent(keys[0] as never)

    expect(pasted.keypress.isPasted).toBe(true)
    expect(pasted.isControlByteChord).toBe(false)
  })
})
