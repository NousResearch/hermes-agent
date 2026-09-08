import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'

// Regression for #103885: kitty CSI-u sequences carrying colon sub-parameters
// (progressive-enhancement flags above 1) must still resolve to a keypress,
// not be swallowed by the CSI-u branch with an empty name. The parser skips
// the sub-parameter and consumes only the leading number of each field,
// mirroring hermes_cli/curses_ui.py::_parse_csi_u_key so the TS and Python
// input paths agree.
describe('CSI-u tolerates kitty colon sub-parameters (#103885)', () => {
  it.each([
    // Standard Shift+a (codepoint 97 = 'a', modifier 2 = shift) — must keep working.
    { seq: '\x1b[97;2u', name: 'a', shift: true },
    // Alternate-key sub-parameter on the codepoint: ESC[97:65;2u — drop :65.
    { seq: '\x1b[97:65;2u', name: 'a', shift: true },
    // Event-type sub-parameter on the modifier: ESC[97;2:1u — drop :1.
    { seq: '\x1b[97;2:1u', name: 'a', shift: true },
    // Sub-parameter on the codepoint with no modifier field: ESC[97:65u.
    { seq: '\x1b[97:65u', name: 'a', shift: false }
  ])('parses $seq as name=$name shift=$shift', ({ seq, name, shift }) => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, seq)

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ name, shift, meta: false, ctrl: false, super: false })
  })

  it('still parses plain CSI-u without sub-parameters (no regression)', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[13;2u')

    expect(keys).toEqual([expect.objectContaining({ name: 'return', shift: true })])
  })
})
