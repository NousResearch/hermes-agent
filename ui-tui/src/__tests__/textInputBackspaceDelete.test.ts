import { describe, expect, it } from 'vitest'

import { backspaceDelete } from '../components/textInput.js'

const KO_KAI = '\u0e01' // ก
const SARA_I = '\u0e34' // ◌ิ
const MAI_THO = '\u0e49' // ◌้

describe('backspaceDelete', () => {
  it('deletes exactly the trailing Thai combining mark at end-of-input', () => {
    const value = KO_KAI + SARA_I

    expect(backspaceDelete(value, value.length)).toEqual({ cursor: 1, value: KO_KAI })
  })

  it('peels trailing combining marks one code point at a time', () => {
    const value = KO_KAI + SARA_I + MAI_THO
    const afterOne = backspaceDelete(value, value.length)

    expect(afterOne).toEqual({ cursor: 2, value: KO_KAI + SARA_I })
    expect(backspaceDelete(afterOne.value, afterOne.cursor)).toEqual({ cursor: 1, value: KO_KAI })
  })

  it('handles combining vowel signs outside the Thai block', () => {
    const value = '\u0915\u093f' // क + DEVANAGARI VOWEL SIGN I

    expect(backspaceDelete(value, value.length)).toEqual({ cursor: 1, value: '\u0915' })
  })

  it('does not split a trailing surrogate pair', () => {
    expect(backspaceDelete('a😀', 'a😀'.length)).toEqual({ cursor: 1, value: 'a' })
  })

  it('retains whole-grapheme Backspace in mid-text, even after a combining mark', () => {
    const value = KO_KAI + SARA_I + KO_KAI + SARA_I

    expect(backspaceDelete(value, 2)).toEqual({ cursor: 0, value: KO_KAI + SARA_I })
  })
})
