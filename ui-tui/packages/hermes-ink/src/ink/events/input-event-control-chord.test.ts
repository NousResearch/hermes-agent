import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

// Regression: #115284 — a stray `l` in the TUI composer after a session
// resume / tab switch / window restore. The dashboard writes the PTY
// force-redraw byte (Ctrl+L, 0x0c — `hermes_cli/pty_session.py`
// ``TUI_FORCE_REDRAW``) into the TUI's stdin on every re-attach
// (`hermes_cli/web_routers/chat_ws.py`: ``session.attach(ws, force_redraw=not
// _created)``). parse-keypress names the byte after the letter it encodes and
// InputEvent hands that name to ``input`` so bindings can match ctrl+<letter>,
// so the byte reached the composer looking like typed text.

const PRINTABLE = /^[ -~\u00a0-\uffff]+$/

/** Mirror the composer's insert gate (ui-tui/src/components/textInput.tsx). */
const composerWouldInsert = (event: InputEvent): boolean =>
  !event.isControlChord && (event.keypress.isPasted || event.input.length > 0) && PRINTABLE.test(event.input)

function parseOne(bytes: string): InputEvent {
  const [keys] = parseMultipleKeypresses({ ...INITIAL_STATE }, bytes)

  return new InputEvent(keys[0] as never)
}

describe('ctrl chords are bindable but never typed text (#115284)', () => {
  it('keeps the binding name on input for the redraw byte and its kitty twin, and refuses to insert either', () => {
    for (const bytes of ['\x0c', '\x1b[108;5u']) {
      const event = parseOne(bytes)

      expect(event.key.ctrl, JSON.stringify(bytes)).toBe(true)
      expect(event.input, JSON.stringify(bytes)).toBe('l') // binding name Ink derives from the chord
      expect(event.isControlChord, JSON.stringify(bytes)).toBe(true)
      expect(composerWouldInsert(event), JSON.stringify(bytes)).toBe(false)
    }
  })

  it('still inserts a typed l and a bracketed paste carrying a control byte', () => {
    const typed = parseOne('l')

    expect(typed.isControlChord).toBe(false)
    expect(composerWouldInsert(typed)).toBe(true)

    const [keys] = parseMultipleKeypresses({ ...INITIAL_STATE }, '\x1b[200~\x0c\x1b[201~')
    const pasted = new InputEvent(keys[0] as never)

    expect(pasted.keypress.isPasted).toBe(true)
    expect(pasted.isControlChord).toBe(false)
  })
})
