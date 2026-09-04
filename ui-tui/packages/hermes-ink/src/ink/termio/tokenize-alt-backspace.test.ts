import { describe, expect, it } from 'vitest'

import { parseMultipleKeypresses } from '../parse-keypress.js'
import { createTokenizer } from './tokenize.js'

const INITIAL = { mode: 'NORMAL', incomplete: '' } as any

describe('legacy Alt+Backspace (ESC + DEL/BS) — #90061 Bug 2', () => {
  it.each(['\x7f', '\x08'])('tokenizer keeps ESC+%j as a single sequence token (one feed)', byte => {
    // xterm-class terminals and macOS Terminal.app encode Alt+Backspace as
    // ESC followed by DEL (0x7f) or BS (0x08). It must stay one sequence
    // token so parse-keypress.ts can preserve the Alt/meta flag — otherwise
    // it splits into Escape + plain Backspace and word-delete breaks.
    const t = createTokenizer({ legacyAltEnter: true })
    const sequence = `\x1b${byte}`

    expect(t.feed(sequence)).toEqual([{ type: 'sequence', value: sequence }])
    expect(t.buffer()).toBe('')
  })

  it.each(['\x7f', '\x08'])('tokenizer reassembles ESC+%j split across two feeds', byte => {
    const t = createTokenizer({ legacyAltEnter: true })
    const sequence = `\x1b${byte}`

    expect(t.feed('\x1b')).toEqual([])
    expect(t.feed(byte)).toEqual([{ type: 'sequence', value: sequence }])
    expect(t.buffer()).toBe('')
  })

  it('tokenizer does NOT merge ESC+DEL when legacyAltEnter is off (output-stream safety)', () => {
    // The output-side Parser creates its tokenizer without legacyAltEnter;
    // there ESC+DEL must stay text and never be reinterpreted as a key.
    const t = createTokenizer()
    expect(t.feed('\x1b\x7f')).toEqual([{ type: 'text', value: '\x1b\x7f' }])
  })

  it.each(['\x7f', '\x08'])('parser yields backspace with meta=true for ESC+%j', byte => {
    const [parsed] = parseMultipleKeypresses(INITIAL, `\x1b${byte}`)
    expect(parsed).toHaveLength(1)

    const key = (parsed[0] as any).key ?? parsed[0]
    expect(key.name).toBe('backspace')
    expect(key.meta).toBe(true)
  })

  it('parser still splits a plain DEL (no ESC) as an unmodified backspace', () => {
    const [parsed] = parseMultipleKeypresses(INITIAL, '\x7f')
    expect(parsed).toHaveLength(1)

    const key = (parsed[0] as any).key ?? parsed[0]
    expect(key.name).toBe('backspace')
    expect(key.meta).toBe(false)
  })

  it('parser still treats a flushed bare Escape as escape (no modifier)', () => {
    // A lone ESC is held in the tokenizer buffer as an incomplete sequence
    // (it may be the prefix of Alt+key). Only a flush forces it out as a key.
    const [, state] = parseMultipleKeypresses(INITIAL, '\x1b')
    const [parsed] = parseMultipleKeypresses(state, null)

    expect(parsed).toHaveLength(1)

    const key = (parsed[0] as any).key ?? parsed[0]
    expect(key.name).toBe('escape')
    expect(key.meta).toBe(false)
  })
})
