import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { reorderBidi } from './bidi.js'
import Output from './output.js'
import { cellAt, CharPool, createScreen, HyperlinkPool, StylePool } from './screen.js'

type BidiCharacter = Parameters<typeof reorderBidi>[0][number]

const charactersFrom = (text: string): BidiCharacter[] =>
  Array.from(new Intl.Segmenter().segment(text), ({ segment: value }, styleId) => ({
    value,
    width: value === '😀' ? 2 : 1,
    styleId,
    hyperlink: `https://example.com/${styleId}`
  }))

const textFrom = (characters: BidiCharacter[]) => characters.map(character => character.value).join('')

beforeEach(() => {
  // Select an existing software-bidi terminal, without faking the host OS.
  vi.stubEnv('TERM_PROGRAM', 'vscode')
})

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('software bidi glyph mirroring', () => {
  it.each([
    ['Hebrew parentheses', '(אבג)', '(גבא)'],
    ['Arabic brackets', 'مرحبا [عالم]', '[ملاع] ابحرم'],
    ['nested brackets', 'אבג [(דה)]', '[(הד)] גבא'],
    ['LTR island in RTL prose', 'אבג (React)', '(React) גבא'],
    ['RTL island in LTR prose', 'English (אבג)', 'English (גבא)'],
    ['supplementary character before brackets', 'אב 😀 (גד)', '(דג) 😀 בא'],
    ['combining mark attached to a bracket', '(\u0301אב)', '(בא)\u0301']
  ])('renders %s', (_name, source, expected) => {
    const input = charactersFrom(source)
    const original = input.map(character => ({ ...character }))

    expect(textFrom(reorderBidi(input))).toBe(expected)
    expect(input).toEqual(original)
  })

  it.each(['hello (React) [42]', '中文 (日本語)', ''])('preserves LTR/no-text identity: %s', source => {
    const input = charactersFrom(source)

    expect(reorderBidi(input)).toBe(input)
  })

  it('preserves cluster metadata and does not mutate mirrored source objects', () => {
    const input = charactersFrom('(אב)')
    const original = input.map(character => ({ ...character }))
    const output = reorderBidi(input)

    expect(output[0]).toEqual({ ...input[3], value: '(' })
    expect(output[3]).toEqual({ ...input[0], value: ')' })
    expect(output[0]).not.toBe(input[3])
    expect(output[3]).not.toBe(input[0])
    expect(output[1]).toBe(input[2])
    expect(output[2]).toBe(input[1])
    expect(input).toEqual(original)
  })

  it('writes mirrored, styled brackets through the real Output screen path', () => {
    const stylePool = new StylePool()
    const screen = createScreen(20, 1, stylePool, new CharPool(), new HyperlinkPool())
    const output = new Output({ width: 20, height: 1, screen, stylePool })
    const source = '\x1b[31m(אבג)\x1b[0m'

    output.write(0, 0, source)
    const rendered = output.get()
    const cells = Array.from({ length: 5 }, (_, x) => cellAt(rendered, x, 0)!)

    expect(cells.map(cell => cell.char).join('')).toBe('(גבא)')
    expect(cells.every(cell => cell.styleId === cells[0]!.styleId)).toBe(true)
    expect(cells[0]!.styleId).not.toBe(stylePool.none)

    // Repainting must not mirror the already-cached visual clusters again.
    output.write(0, 0, source)
    const repainted = output.get()

    expect(Array.from({ length: 5 }, (_, x) => cellAt(repainted, x, 0)!.char).join('')).toBe('(גבא)')
  })
})
