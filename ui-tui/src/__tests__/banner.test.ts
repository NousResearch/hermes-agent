import { describe, expect, it } from 'vitest'

import { parseRichMarkup } from '../banner.js'

describe('parseRichMarkup', () => {
  it('keeps every segment of each source line on that line’s row', () => {
    // Regression: the old shape returned one entry PER SEGMENT, so art like
    // `[#a1]W[/][#b2]e[/]...` rendered one character per terminal row.
    const rows = parseRichMarkup(
      '[#ff0000]a[/][#00ff00]b[/][#0000ff]c[/]\nhello [#ff0000]world[/] bye'
    )

    expect(rows).toEqual([
      [
        ['#ff0000', 'a'],
        ['#00ff00', 'b'],
        ['#0000ff', 'c']
      ],
      [
        ['', 'hello '],
        ['#ff0000', 'world'],
        ['', ' bye']
      ]
    ])
  })

  it('carries a span across line boundaries instead of leaking raw tags', () => {
    const rows = parseRichMarkup('[#ff0000]ab\ncd[/]')

    expect(rows).toEqual([[['#ff0000', 'ab']], [['#ff0000', 'cd']]])
  })
})
