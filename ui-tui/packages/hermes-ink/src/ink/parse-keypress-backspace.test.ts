import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'
import { PASTE_END, PASTE_START } from './termio/csi.js'
import { createTokenizer } from './termio/tokenize.js'

describe('legacy Alt+Backspace input boundaries', () => {
  it.each(['\x7f', '\b'])('preserves ESC+%j in complete and split reads', backspace => {
    const sequence = `\x1b${backspace}`

    for (const reads of [[sequence], ['\x1b', backspace]]) {
      let state = { ...INITIAL_STATE }

      const keys = reads.flatMap(read => {
        const [parsed, next] = parseMultipleKeypresses(state, read)
        state = next

        return parsed
      })

      expect(keys).toEqual([
        expect.objectContaining({ kind: 'key', name: 'backspace', meta: true, ctrl: false, super: false, sequence })
      ])
      expect(parseMultipleKeypresses(state, null)[0]).toEqual([])
    }
  })

  it.each(['\x7f', '\b'])('does not merge a flushed Escape with later %j', backspace => {
    const [pending, state] = parseMultipleKeypresses(INITIAL_STATE, '\x1b')
    expect(pending).toEqual([])
    const [escape, flushed] = parseMultipleKeypresses(state, null)
    expect(escape).toEqual([expect.objectContaining({ name: 'escape', meta: false })])
    expect(parseMultipleKeypresses(flushed, backspace)[0]).toEqual([
      expect.objectContaining({ name: 'backspace', meta: false })
    ])
  })

  it.each(['\x7f', '\b'])('preserves surrounding text and bracketed paste for %j', backspace => {
    const sequence = `\x1b${backspace}`
    expect(parseMultipleKeypresses(INITIAL_STATE, `ab${sequence}ô`)[0]).toEqual([
      expect.objectContaining({ raw: 'ab' }),
      expect.objectContaining({ name: 'backspace', meta: true }),
      expect.objectContaining({ raw: 'ô' })
    ])
    expect(parseMultipleKeypresses(INITIAL_STATE, PASTE_START + sequence + PASTE_END)[0]).toEqual([
      expect.objectContaining({ isPasted: true, raw: sequence })
    ])
  })

  it.each(['\x1b[127;3u', '\x1b[27;3;127~'])('keeps extended encoding %j intact at every read boundary', sequence => {
    for (let split = 1; split < sequence.length; split++) {
      const [first, state] = parseMultipleKeypresses(INITIAL_STATE, sequence.slice(0, split))
      const [last] = parseMultipleKeypresses(state, sequence.slice(split))
      expect([...first, ...last]).toEqual([expect.objectContaining({ name: 'backspace', meta: true, super: false })])
    }
  })

  it.each(['\x7f', '\b'])('leaves output-tokenizer control text unchanged for %j', backspace => {
    const tokenizer = createTokenizer()
    expect(tokenizer.feed(`\x1b${backspace}`)).toEqual([{ type: 'text', value: `\x1b${backspace}` }])
  })
})
