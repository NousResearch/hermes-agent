import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'

// Probe: feed many exotic IME-ish byte patterns straight through the parser
// and assert NO printable character codepoint silently vanishes. This catches
// the "chunk falls through every branch -> name:'' with a non-printable
// sequence -> composer discards it" failure class for sequences we haven't
// hand-enumerated.

function keysToText(keys: Array<{ name?: string; sequence?: string }>): string {
  // Reconstruct what the composer would insert: backspaces delete, everything
  // else with a printable sequence inserts its sequence.
  let out = ''
  for (const k of keys) {
    if (k.name === 'backspace') {
      out = out.slice(0, -1)
      continue
    }
    const seq = k.sequence ?? ''
    // Mirror the composer's PRINTABLE gate
    if (/^[ -~\u00a0-\uffff]+$/.test(seq)) {
      out += seq
    } else if (seq) {
      // Non-printable, non-backspace => composer drops it. Mark it so the
      // assertion can show what was lost.
      out += `«DROP:${[...seq].map(c => 'U+' + c.codePointAt(0)!.toString(16)).join(',')}»`
    }
  }
  return out
}

const cases: Array<[string, string, string]> = [
  // [label, input bytes, expected text after composer-emulation]
  ['fused bs+char', '\x7fô', 'ô'], // starts empty, bs no-ops in our emul
  ['fused bs+2char', '\x7fôi', 'ôi'],
  ['embedded bs', 'ab\bç', 'aç'],
  ['hard-erase \\b \\b + char', '\b \bô', 'ô'],
  ['hard-erase x3 + ạnh (from anh)', 'anh\b \b\b \b\b \bạnh', 'ạnh'],
  ['DEL-space-DEL + char', '\x7f \x7fô', 'ô'],
  ['trailing text after multi DEL', '\x7f\x7f\x7fươn', 'ươn'],
  ['char then DEL then char fused', 'o\x7fô', 'ô'],
  ['multiple syllable fused', 'vuon\x7f\x7f\x7fương', 'vương'],
  // CR/LF are intentionally NOT split (preserve paste/return semantics), so a
  // text token with an embedded CR is left whole; assert it is NOT split into
  // surviving letters here — that path is covered by the composer's return /
  // paste handling, not parseTextKeypresses.
]

describe('parser does not silently drop printable codepoints', () => {
  for (const [label, input, expected] of cases) {
    it(label, () => {
      const [keys] = parseMultipleKeypresses(INITIAL_STATE, input)
      const text = keysToText(keys as Array<{ name?: string; sequence?: string }>)
      expect(text, `keys=${JSON.stringify(keys)}`).toBe(expected)
    })
  }

  it('exhaustive: DEL between every pair of letters never drops a letter', () => {
    const letters = [...'aăâeêioôơuưy']
    for (const a of letters) {
      for (const b of letters) {
        const input = `${a}\x7f${b}`
        const [keys] = parseMultipleKeypresses(INITIAL_STATE, input)
        const text = keysToText(keys as Array<{ name?: string; sequence?: string }>)
        // a inserted, bs removes a, b inserted => "b"
        expect(text, `input=${JSON.stringify(input)} keys=${JSON.stringify(keys)}`).toBe(b)
      }
    }
  })
})
