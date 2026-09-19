import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from '../../packages/hermes-ink/src/ink/parse-keypress.js'

import { shouldInsertText } from '../components/textInput.js'

// Regression for the stray-l bug (#87069): input-event.ts maps ctrl-modified
// keys to the bare letter name, so unbound ctrl+letter chords — the Ctrl+L
// redraw, and the 0x0c the dashboard injects on PTY reattach — reached the
// composer's insert fallthrough as "printable" single chars and typed
// themselves into the input box. shouldInsertText is the gate that drops
// them while keeping plain typing and pastes.

type Parsed = { ctrl: boolean; name?: string; sequence?: string }

/** Mirror of input-event.ts's mapping: what the composer sees as `input`. */
const composerInput = (kp: Parsed): string => (kp.ctrl ? kp.name ?? '' : kp.sequence ?? '')

describe('shouldInsertText', () => {
  it('drops the ctrl+L redraw byte (0x0c) the way the composer receives it', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x0c')
    const kp = keys[0] as unknown as Parsed
    expect(kp.ctrl).toBe(true)
    // the load-bearing upstream mapping: ctrl+letter arrives named as its letter
    expect(composerInput(kp)).toBe('l')
    expect(shouldInsertText(kp, false, composerInput(kp))).toBe(false)
  })

  it('drops every unbound ctrl+letter chord, not just l', () => {
    for (const letter of 'glnpqrs') {
      const [keys] = parseMultipleKeypresses(INITIAL_STATE, String.fromCharCode(letter.charCodeAt(0) - 96))
      const kp = keys[0] as unknown as Parsed
      expect(composerInput(kp)).toBe(letter)
      expect(shouldInsertText(kp, false, composerInput(kp))).toBe(false)
    }
  })

  it('keeps plain typed letters', () => {
    expect(shouldInsertText({ ctrl: false }, false, 'x')).toBe(true)
  })

  it('keeps pastes regardless of modifier bits', () => {
    expect(shouldInsertText({ ctrl: true }, true, 'l')).toBe(true)
  })
})
