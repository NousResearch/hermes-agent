import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

import { reorderBidi } from './bidi.js'
import Output from './output.js'
import { cellAt, CharPool, createScreen, HyperlinkPool, StylePool } from './screen.js'

type ClusteredChar = Parameters<typeof reorderBidi>[0][number]

const cluster = (value: string, overrides: Partial<ClusteredChar> = {}): ClusteredChar => ({
  value,
  width: 1,
  styleId: 0,
  hyperlink: undefined,
  ...overrides
})

function renderLine(text: string): ReturnType<typeof cellAt>[] {
  const width = 32
  const stylePool = new StylePool()
  const screen = createScreen(width, 1, stylePool, new CharPool(), new HyperlinkPool())
  const output = new Output({ width, height: 1, stylePool, screen })

  output.write(0, 0, text)
  output.get()
  const rendered = output.get()

  return Array.from({ length: width }, (_, x) => cellAt(rendered, x, 0))
}

describe('software bidi glyph mirroring', () => {
  beforeAll(() => {
    vi.stubEnv('TERM_PROGRAM', 'vscode')
  })

  afterAll(() => {
    vi.unstubAllEnvs()
  })

  it.each([
    ['(אבג)', '(גבא)'],
    ['مرحبا [عالم]', '[ملاع] ابحرم'],
    ['אבג (React)', '(React) גבא'],
    ['[אב(ג)]', '[(ג)בא]'],
    ['∈אבג', 'גבא∋']
  ])('renders mapped punctuation through Output: %s', (logical, visual) => {
    const cells = renderLine(logical)

    expect(
      cells
        .slice(0, visual.length)
        .map(cell => cell?.char)
        .join('')
    ).toBe(visual)
  })

  it('preserves style and hyperlink metadata on mirrored clusters', () => {
    const cells = renderLine('\u001B[31m\u001B]8;;https://example.com\u0007(\u001B]8;;\u0007\u001B[0mאבג)')
    const first = cells[0]!
    const last = cells[4]!

    expect(first?.char).toBe('(')
    expect(first?.hyperlink).toBeUndefined()
    expect(last?.char).toBe(')')
    expect(last?.hyperlink).toBe('https://example.com')
    expect(last?.styleId).not.toBe(first?.styleId)
  })

  it('accounts for supplementary-plane offsets without mutating logical clusters', () => {
    const characters = [
      cluster('('),
      cluster('א'),
      cluster('ב'),
      cluster('😀', { width: 2 }),
      cluster('ג'),
      cluster(')')
    ]

    const snapshot = characters.map(character => ({ ...character }))
    const reordered = reorderBidi(characters)

    expect(reordered.map(character => character.value).join('')).toBe('(ג😀בא)')
    expect(characters).toEqual(snapshot)
    expect(reordered[1]).toBe(characters[4])
    expect(reordered[2]).toBe(characters[3])
    expect(reordered[0]).not.toBe(characters[5])
    expect(reordered[0]).toMatchObject({ ...characters[5], value: '(' })
  })

  it('keeps the pure-LTR identity fast path', () => {
    const characters = [cluster('a'), cluster('('), cluster('b'), cluster(')')]

    expect(reorderBidi(characters)).toBe(characters)
  })
})
