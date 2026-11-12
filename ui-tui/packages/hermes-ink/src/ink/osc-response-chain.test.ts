import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'
import { oscColor, TerminalQuerier } from './terminal-querier.js'
import { parseOscColor } from './terminal.js'

// End-to-end (minus the PTY): a terminal's OSC 11 reply arriving on stdin
// must tokenize as a response, match the pending oscColor(11) query, and
// parse to a hex background. This is the exact chain App.tsx relies on for
// light/dark detection — if any segment drops the reply, detection dies
// silently because the querier never times out.

const parseWire = (raw: string) => {
  const [keys] = parseMultipleKeypresses({ ...INITIAL_STATE }, raw)

  return keys
}

const drainResponses = (raw: string) => parseWire(raw).filter(k => k.kind === 'response')

describe('OSC 11 reply chain', () => {
  it.each([
    ['BEL-terminated', '\x1b]11;rgb:ffff/ffff/ffff\x07', '#ffffff'],
    ['ST-terminated', '\x1b]11;rgb:1e1e/1e1e/2e2e\x1b\\', '#1e1e2e']
  ])('resolves a pending query from a %s reply', async (_label, wire, expectedHex) => {
    const writes: string[] = []
    const stdout = { write: (s: string) => (writes.push(s), true) } as unknown as NodeJS.WriteStream
    const querier = new TerminalQuerier(stdout)

    const pending = querier.send(oscColor(11))

    expect(writes.join('')).toContain('\x1b]11;?')

    const responses = drainResponses(wire)

    expect(responses).toHaveLength(1)

    for (const r of responses) {
      if (r.kind === 'response') {
        querier.onResponse(r.response)
      }
    }

    const reply = await pending

    expect(reply?.type).toBe('osc')

    if (reply?.type === 'osc') {
      expect(reply.code).toBe(11)
      expect(parseOscColor(reply.data)).toBe(expectedHex)
    }
  })

  it('reply interleaved with keystrokes still resolves', async () => {
    const stdout = { write: () => true } as unknown as NodeJS.WriteStream
    const querier = new TerminalQuerier(stdout)
    const pending = querier.send(oscColor(11))

    const keys = parseWire('a\x1b]11;rgb:0000/0000/0000\x07b')

    for (const k of keys) {
      if (k.kind === 'response') {
        querier.onResponse(k.response)
      }
    }

    const reply = await pending

    expect(reply?.type).toBe('osc')

    // The surrounding keystrokes survive as normal input.
    const chars = keys.filter(k => k.kind !== 'response')

    expect(chars.length).toBeGreaterThanOrEqual(2)
  })
})
