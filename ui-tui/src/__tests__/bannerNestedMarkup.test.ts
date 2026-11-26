import { describe, expect, it } from 'vitest'

import { parseRichMarkup } from '../banner.js'

describe('parseRichMarkup', () => {
  it('keeps a nested colour tag as a colour instead of literal text', () => {
    // A custom banner_logo/hero using [#outer][#inner]text[/]text[/] used to render
    // the inner tag as visible text: the regex matched up to the FIRST [/].
    expect(parseRichMarkup('[#111111][#222222]inner[/]outer[/]')).toEqual([
      [
        ['#222222', 'inner'],
        ['#111111', 'outer'],
      ],
    ])
  })

  it('leaves untagged text on the default colour', () => {
    expect(parseRichMarkup('hello')).toEqual([[['', 'hello']]])
  })

  it('parses parallel (non-nested) tags on one line', () => {
    expect(parseRichMarkup('[#111111]a[/][#222222]b[/]')).toEqual([
      [
        ['#111111', 'a'],
        ['#222222', 'b'],
      ],
    ])
  })

  it('keeps lines separate and pads a blank line', () => {
    expect(parseRichMarkup('a\n\nb')).toEqual([[['', 'a']], [['', ' ']], [['', 'b']]])
  })

  it('does not lose text after an unbalanced closing tag', () => {
    // The stray [/] splits the line but must not swallow either side.
    expect(parseRichMarkup('stray[/]text')).toEqual([
      [
        ['', 'stray'],
        ['', 'text'],
      ],
    ])
  })
})
