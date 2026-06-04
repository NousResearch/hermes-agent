import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, type ParsedInput, type ParsedKey, parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

/**
 * Drive the real input pipeline (tokenizer → parseKeypress → InputEvent) for a
 * sequence of stdin chunks. `null` chunks simulate App's 50ms flush watchdog
 * firing mid-sequence. Returns the `.input` of the first key-kind token — i.e.
 * what would actually be typed into the composer.
 */
function pipelineInput(...chunks: (string | null)[]): string {
  let state = INITIAL_STATE
  const all: ParsedInput[] = []

  for (const chunk of chunks) {
    const [keys, next] = parseMultipleKeypresses(state, chunk)
    all.push(...keys)
    state = next
  }

  const key = all.find((k): k is ParsedKey => k.kind === 'key')

  return key ? new InputEvent(key).input : ''
}

describe('InputEvent SGR mouse fragment suppression', () => {
  it('suppresses the buffered CSI prefix force-emitted by a mid-sequence flush', () => {
    // The tokenizer buffers an incomplete CSI mouse sequence; the flush
    // force-emits it as a nameless sequence token (ESC still attached). Intact
    // `[<btn;col;row M` sequences are recovered as mouse/wheel events upstream,
    // so only these terminatorless prefixes fall through to the guard.
    expect(pipelineInput('\x1b[<0;35;', null)).toBe('')
    expect(pipelineInput('\x1b[<0;35;46', null)).toBe('')
  })

  it('suppresses 1-, 2-, and 3-field ESC-less continuation tails', () => {
    // These are the cases the older `/^\[<\d+;\d+;\d+[Mm]/` guard missed —
    // the prefix was lost to the flush, only the tail reaches us as text.
    for (const tail of ['46M', '6M', '35;46M', '0;35;46M']) {
      expect(pipelineInput(tail)).toBe('')
    }
  })

  it('suppresses leading-semicolon tails from a split at a `;` boundary', () => {
    for (const tail of [';46M', ';35;46M']) {
      expect(pipelineInput(tail)).toBe('')
    }
  })

  it('suppresses both halves of a `ESC[<0; / 35;46M` split end to end', () => {
    expect(pipelineInput('\x1b[<0;', null)).toBe('') // flushed prefix
    expect(pipelineInput('35;46M')).toBe('') // continuation
  })

  it('suppresses release (`m`) terminators as well as press (`M`)', () => {
    expect(pipelineInput('35;46m')).toBe('')
    expect(pipelineInput('\x1b[<0;35;', null)).toBe('')
  })
})

describe('InputEvent SGR mouse fragment guard does not eat real input', () => {
  it('passes through lone bracket/angle/semicolon characters', () => {
    // No coordinate digit → the `(?=…\d)` lookahead fails, so typing these
    // characters is never swallowed.
    expect(pipelineInput('<')).toBe('<')
    expect(pipelineInput('[')).toBe('[')
    expect(pipelineInput(';')).toBe(';')
  })

  it('passes through digits and the literal letter M', () => {
    // These parse to a named key (number / m), so the `!keypress.name` gate
    // skips suppression entirely.
    expect(pipelineInput('5')).toBe('5')
    expect(pipelineInput('M')).toBe('M')
  })

  it('passes through ordinary text', () => {
    expect(pipelineInput('hello')).toBe('hello')
  })

  it('keeps two stuck-together fragments / coordinate-like prose intact', () => {
    // An embedded M/m breaks the `[\d;]+...$` anchor, so a run like this is
    // left for the upstream burst/recovery logic rather than blanked here.
    expect(pipelineInput('1234;56;78M9;10;11M')).toBe('1234;56;78M9;10;11M')
  })
})
