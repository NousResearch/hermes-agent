import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'

/**
 * Probe the various byte patterns a mobile Android terminal can deliver for
 * a single Enter, to expose gaps in the current CRLF fix.
 */
describe('mobile Enter arrival patterns (analysis probe)', () => {
  it('lone CR', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\r')
    expect(keys.map(k => k.name)).toEqual(['return'])
  })

  it('lone LF', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\n')
    expect(keys.map(k => k.name)).toEqual(['return'])
  })

  it('CRLF fused in one read', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\r\n')
    expect(keys.map(k => k.name)).toEqual(['return'])
  })

  it('LFCR fused in one read', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\n\r')
    expect(keys.map(k => k.name)).toEqual(['return'])
  })

  it('CR then LF in separate reads: two Enters (handled upstream, not here)', () => {
    // A split CRLF (CR in one frame, LF in the next) must NOT be fused at the
    // parse-keypress layer: the backend cannot know whether a lone CR is the
    // first half of a CRLF or a real Enter, and holding it would delay real
    // Enters until the next read (a composer-submit regression). The browser
    // web dashboard fuses the split CRLF BEFORE it reaches the PTY (see
    // createPtyEnterFuser in web/src/lib/pty-mobile-input.ts), so the PTY
    // never sees the halves separately. Backend-parse therefore emits two
    // 'return' keys for two lone line-ending reads.
    const [k1, s1] = parseMultipleKeypresses(INITIAL_STATE, '\r')
    const [k2] = parseMultipleKeypresses(s1, '\n')
    expect([...k1.map(k => k.name), ...k2.map(k => k.name)]).toEqual(['return', 'return'])
  })

  it('LF then CR in separate reads: two Enters (handled upstream, not here)', () => {
    const [k1, s1] = parseMultipleKeypresses(INITIAL_STATE, '\n')
    const [k2] = parseMultipleKeypresses(s1, '\r')
    expect([...k1.map(k => k.name), ...k2.map(k => k.name)]).toEqual(['return', 'return'])
  })

  it('CRLF embedded in text stays whole (paste)', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, 'a\r\nb')
    expect(keys.map(k => k.name)).toEqual([''])
  })
})

/**
 * Sticky-modifier class: Android/mobile terminals deliver a modified Enter as
 * ESC followed by the CRLF newline halves. The tokenizer splits that into an
 * ESC+CR / ESC+LF sequence (an Alt+Enter) plus a separate opposite line-ending
 * text token, which would otherwise re-parse as a second bare Enter and submit
 * the composer twice. The parser fuses the halves into one modified Enter.
 */
describe('sticky-modifier ESC + CRLF fused (modified Enter)', () => {
  it('ESC CRLF fused in one read is one Alt+Enter', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b\r\n')
    expect(keys.map(k => k.name)).toEqual(['return'])
    expect(keys[0]).toMatchObject({ name: 'return', meta: true })
  })

  it('ESC LFCR fused in one read is one Alt+Enter', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b\n\r')
    expect(keys.map(k => k.name)).toEqual(['return'])
    expect(keys[0]).toMatchObject({ name: 'return', meta: true })
  })

  it('a lone ESC+CR followed by a real separate Enter stays two Enters', () => {
    // No fusion across a real boundary: ESC+CR is one Alt+Enter, then a fresh
    // bare Enter on the same read is a second keypress (not the CRLF half).
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b\r\r')
    expect(keys.map(k => k.name)).toEqual(['return', 'return'])
  })

  it('ESC+CR followed by text is one Alt+Enter plus the text', () => {
    // The next token is not the opposite line-ending, so nothing is swallowed.
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b\ra')
    expect(keys.map(k => k.name)).toEqual(['return', 'a'])
  })

  it('a lone ESC+CR alone (no CRLF half) stays one Alt+Enter', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b\r')
    expect(keys.map(k => k.name)).toEqual(['return'])
    expect(keys[0]).toMatchObject({ name: 'return', meta: true })
  })
})
