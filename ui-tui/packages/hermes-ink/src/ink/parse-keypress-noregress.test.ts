import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'

// Confirm the control-byte split is a NO-OP for clean input (EVKey-style:
// backspace and recomposed text arrive in separate, non-fused reads). The
// fix must not change behavior for any text token that has no embedded
// control byte — otherwise it could regress IMEs that already work.
describe('control-byte split does not touch clean (EVKey-style) input', () => {
  it('a plain printable text token yields exactly one keypress (no spurious split)', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'ạnh')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ raw: 'ạnh' })
  })

  it('a lone backspace read is unchanged', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x7f')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ name: 'backspace' })
  })

  it('separate clean reads (bs read, then text read) each produce one key', () => {
    const [k1] = parseMultipleKeypresses(INITIAL_STATE, '\x7f')
    const [k2] = parseMultipleKeypresses(INITIAL_STATE, 'ô')

    expect(k1).toHaveLength(1)
    expect(k1[0]).toMatchObject({ name: 'backspace' })
    expect(k2).toHaveLength(1)
    expect(k2[0]).toMatchObject({ raw: 'ô' })
  })

  it('a full clean Vietnamese word with no embedded control bytes is one text key', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'vương')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ raw: 'vương' })
  })
})
