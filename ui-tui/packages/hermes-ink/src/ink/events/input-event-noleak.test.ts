import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

/** Feed one raw sequence through the full tokenizer → InputEvent pipeline. */
const inputForSequence = (sequence: string): string => {
  const [keys] = parseMultipleKeypresses(INITIAL_STATE, sequence)

  expect(keys).toHaveLength(1)

  return new InputEvent(keys[0]!).input
}

describe('modifyOtherKeys named keys produce no text', () => {
  // Regression: Alt+Backspace on Ghostty arrives as ESC[27;3;127~. The old
  // inputForSpecialSequence fell through to `name`, so once the input line was
  // empty the literal word "backspace" leaked into the prompt.
  it('Alt+Backspace via modifyOtherKeys leaks no text', () => {
    expect(inputForSequence('\x1b[27;3;127~')).toBe('')
  })

  it('Ctrl+Backspace via modifyOtherKeys leaks no text', () => {
    expect(inputForSequence('\x1b[27;5;127~')).toBe('')
  })

  it('Alt+Backspace via kitty CSI-u leaks no text', () => {
    expect(inputForSequence('\x1b[127;3u')).toBe('')
  })

  it('plain Backspace still parses without text', () => {
    expect(inputForSequence('\x7f')).toBe('')
  })
})

describe('printable special-sequence forms are preserved', () => {
  it('Alt+a via modifyOtherKeys still types "a"', () => {
    expect(inputForSequence('\x1b[27;3;97~')).toBe('a')
  })

  it('Ctrl+Space via CSI-u still types a space', () => {
    expect(inputForSequence('\x1b[32;5u')).toBe(' ')
  })
})
