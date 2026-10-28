import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from '../../packages/hermes-ink/src/ink/parse-keypress.js'

// Contract for the Alt+arrow turn-jump chords, from the dashboard E2E finding:
// plain `[`/`]` are composer text and ESC-prefixed brackets are swallowed by
// the tokenizer as OSC/CSI sequence starts (they never arrive as keypresses),
// while Alt+arrows decode cleanly to name=left/right with meta — the handler
// matches `key.alt && (leftArrow || rightArrow)`.
describe('Alt+arrow chord decoding (turn-jump contract)', () => {
  it('decodes Alt+→ (xterm) as right arrow keypress', () => {
    const [inputs, state] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[1;3C')
    const [flushed] = parseMultipleKeypresses(state, null)
    const k = [...inputs, ...flushed][0] as { key?: { meta?: boolean; name?: string }; name?: string }

    expect(k.name).toBe('right')
  })

  it('decodes ← (CSI D) as left arrow keypress', () => {
    const [inputs, state] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[D')
    const [flushed] = parseMultipleKeypresses(state, null)
    const k = [...inputs, ...flushed][0] as { key?: { meta?: boolean; name?: string }; name?: string }

    expect(k.name).toBe('left')
  })

  it('ESC+] is swallowed as an OSC start (never a keypress) — plain brackets must stay composer text', () => {
    const [inputs, state] = parseMultipleKeypresses(INITIAL_STATE, '\x1b]')
    const [flushed] = parseMultipleKeypresses(state, null)

    expect([...inputs, ...flushed]).toHaveLength(0)
  })
})
