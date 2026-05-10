import { describe, expect, it } from 'vitest'

import { artWidth, parseRichMarkup } from '../banner.js'

describe('parseRichMarkup', () => {
  it('keeps multiple colored segments on the same rendered line', () => {
    const lines = parseRichMarkup('[#7a2f26]rick[/] [#f0c7a0]sanchez[/]')

    expect(lines).toEqual([
      [
        ['#7a2f26', 'rick'],
        ['', ' '],
        ['#f0c7a0', 'sanchez']
      ]
    ])
    expect(artWidth(lines)).toBe('rick sanchez'.length)
  })

  it('keeps foreground and background colors for dense terminal art', () => {
    const lines = parseRichMarkup('[#f1d7c2 on #5c2419]▀[/]')

    expect(lines).toEqual([[['#f1d7c2', '▀', '#5c2419']]])
    expect(artWidth(lines)).toBe(1)
  })
})
