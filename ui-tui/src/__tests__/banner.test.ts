import { describe, expect, it } from 'vitest'

import { artWidth, parseRichMarkup } from '../banner.js'

describe('parseRichMarkup', () => {
  it('keeps per-character gradient segments on a single row', () => {
    // Regression: the old shape returned one entry PER SEGMENT, so gradient
    // art like `[#a1]W[/][#b2]e[/]...` rendered one character per terminal row.
    const rows = parseRichMarkup('[#ff0000]a[/][#00ff00]b[/][#0000ff]c[/]')

    expect(rows).toEqual([
      [
        ['#ff0000', 'a'],
        ['#00ff00', 'b'],
        ['#0000ff', 'c']
      ]
    ])
  })

  it('keeps every segment of a source line on that line’s row', () => {
    const rows = parseRichMarkup('hello [#ff0000]world[/] bye\nsecond [#00ff00]line[/]')

    expect(rows).toEqual([
      [
        ['', 'hello '],
        ['#ff0000', 'world'],
        ['', ' bye']
      ],
      [
        ['', 'second '],
        ['#00ff00', 'line']
      ]
    ])
  })

  it('carries a span across line boundaries instead of leaking raw tags', () => {
    const rows = parseRichMarkup('[#ff0000]ab\ncd[/]')

    expect(rows).toEqual([[['#ff0000', 'ab']], [['#ff0000', 'cd']]])
  })
})

describe('artWidth', () => {
  it('sums segment widths within each row', () => {
    expect(
      artWidth([
        [
          ['#ff0000', 'ab'],
          ['', 'c']
        ],
        [['', 'xy']]
      ])
    ).toBe(3)
  })
})
