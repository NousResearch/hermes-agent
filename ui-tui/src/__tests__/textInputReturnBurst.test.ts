import { describe, expect, it } from 'vitest'

import { valueForReturnSubmit } from '../components/textInput.js'

describe('valueForReturnSubmit', () => {
  it('includes printable input that arrives in the same keypress as return', () => {
    expect(valueForReturnSubmit('为什么打字上屏，', 8, '会丢失内容')).toEqual({
      cursor: 13,
      value: '为什么打字上屏，会丢失内容'
    })
  })

  it('keeps IME commit text when it arrives in the same burst as return', () => {
    expect(valueForReturnSubmit('为什么打字上屏，', 8, '会丢失内容\r')).toEqual({
      cursor: 13,
      value: '为什么打字上屏，会丢失内容'
    })
  })

  it('leaves the draft unchanged when return carries no printable input', () => {
    expect(valueForReturnSubmit('hello', 5, '')).toEqual({ cursor: 5, value: 'hello' })
    expect(valueForReturnSubmit('hello', 5, '\r')).toEqual({ cursor: 5, value: 'hello' })
    expect(valueForReturnSubmit('hello', 5, '\n')).toEqual({ cursor: 5, value: 'hello' })
  })
})
