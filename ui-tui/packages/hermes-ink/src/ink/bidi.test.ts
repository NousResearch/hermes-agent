import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { firstStrongDirection, reorderBidi, resolveSoftwareBidi, setSoftwareBidiForTesting } from './bidi.js'
import Output from './output.js'
import { cellAt, CellWidth, CharPool, createScreen, HyperlinkPool, StylePool } from './screen.js'
import { stringWidth } from './stringWidth.js'

const WIDTH = 40
const HEIGHT = 4

// Visual order of a run of single-code-point characters at an odd level.
const rev = (s: string) => [...s].reverse().join('')

const CYAN = '\x1b[36m'
const RESET_FG = '\x1b[39m'

function paint(text: string, softWrap?: boolean[]) {
  const stylePool = new StylePool()
  const screen = createScreen(WIDTH, HEIGHT, stylePool, new CharPool(), new HyperlinkPool())
  const output = new Output({ height: HEIGHT, screen, stylePool, width: WIDTH })

  output.write(0, 0, text, softWrap)
  const painted = output.get()

  const cells = (y: number) =>
    Array.from({ length: WIDTH }, (_, x) => cellAt(painted, x, y)!).filter(c => c.width !== CellWidth.SpacerTail)

  const row = (y: number) =>
    cells(y)
      .map(c => c.char)
      .join('')
      .trimEnd()

  const styleOf = (y: number, char: string) => {
    const cell = cells(y).find(c => c.char === char)

    return cell ? stylePool.get(cell.styleId).map(code => code.code) : undefined
  }

  return { row, styleOf }
}

const clustered = (text: string) => [...text].map(value => ({ hyperlink: undefined, styleId: 0, value, width: 1 }))

describe('resolveSoftwareBidi', () => {
  it('is off by default on POSIX terminals, which may do bidi themselves', () => {
    expect(resolveSoftwareBidi({}, 'linux')).toBe(false)
    expect(resolveSoftwareBidi({ TERM: 'xterm-256color' }, 'darwin')).toBe(false)
  })

  it('is on where the host is known to lack bidi', () => {
    expect(resolveSoftwareBidi({}, 'win32')).toBe(true)
    expect(resolveSoftwareBidi({ WT_SESSION: 'x' }, 'linux')).toBe(true)
    expect(resolveSoftwareBidi({ TERM_PROGRAM: 'vscode' }, 'linux')).toBe(true)
  })

  it('lets HERMES_TUI_BIDI force it either way', () => {
    expect(resolveSoftwareBidi({ HERMES_TUI_BIDI: '1' }, 'linux')).toBe(true)
    expect(resolveSoftwareBidi({ HERMES_TUI_BIDI: 'on' }, 'darwin')).toBe(true)
    expect(resolveSoftwareBidi({ HERMES_TUI_BIDI: '0' }, 'win32')).toBe(false)
    expect(resolveSoftwareBidi({ HERMES_TUI_BIDI: 'false', TERM_PROGRAM: 'vscode' }, 'linux')).toBe(false)
  })

  it('ignores an unrecognised HERMES_TUI_BIDI value', () => {
    expect(resolveSoftwareBidi({ HERMES_TUI_BIDI: 'maybe' }, 'linux')).toBe(false)
    expect(resolveSoftwareBidi({ HERMES_TUI_BIDI: '' }, 'win32')).toBe(true)
  })
})

describe('firstStrongDirection', () => {
  it('reads the first strong character', () => {
    expect(firstStrongDirection('مرحبا Hermes')).toBe('rtl')
    expect(firstStrongDirection('10.0.0.138 Hermes مرحبا')).toBe('ltr')
    expect(firstStrongDirection('… 42 !')).toBeUndefined()
  })
})

describe('bidi reordering (software bidi on)', () => {
  beforeEach(() => setSoftwareBidiForTesting(true))
  afterEach(() => setSoftwareBidiForTesting(undefined))

  it('shows Arabic prose in visual order', () => {
    expect(paint('مرحبا بالعالم').row(0)).toBe(rev('مرحبا بالعالم'))
  })

  it('keeps an embedded IP address left-to-right as one run', () => {
    const { row } = paint('العنوان 10.0.0.138')

    expect(row(0)).toBe(`10.0.0.138 ${rev('العنوان')}`)
    expect(row(0)).not.toContain('831.0.0.01')
  })

  it('puts line-end punctuation of an RTL line on the left', () => {
    expect(paint('مرحبا.').row(0)).toBe(`.${rev('مرحبا')}`)
    expect(paint('كيف حالك؟').row(0)).toBe(`؟${rev('كيف حالك')}`)
  })

  it('mirrors brackets inside RTL runs', () => {
    expect(paint('(مثال)').row(0)).toBe(`(${rev('مثال')})`)
  })

  it('orders mixed Arabic and English by the RTL paragraph', () => {
    expect(paint('أنا أستخدم Hermes يوميا').row(0)).toBe(`${rev('يوميا')} Hermes ${rev('أنا أستخدم')}`)
  })

  it('keeps styled inline code as an LTR run with its style', () => {
    const { row, styleOf } = paint(`شغل ${CYAN}npm test${RESET_FG} الآن`)

    expect(row(0)).toBe(`${rev('الآن')} npm test ${rev('شغل')}`)
    expect(styleOf(0, 'n')).toEqual([CYAN])
    expect(styleOf(0, 'ش')).toEqual([])
  })

  it('leaves a pure-LTR line untouched', () => {
    expect(paint('hello (world) 10.0.0.138').row(0)).toBe('hello (world) 10.0.0.138')
  })

  it('carries the paragraph direction onto soft-wrapped continuation lines', () => {
    // Line 2 starts with an English word; on its own it would be an LTR paragraph.
    const wrapped = paint('مرحبا بك\nHermes هو وكيل', [false, true])

    expect(wrapped.row(0)).toBe(rev('مرحبا بك'))
    expect(wrapped.row(1)).toBe(`${rev('هو وكيل')} Hermes`)

    // A hard newline starts a new paragraph, which resolves its own direction.
    const hard = paint('مرحبا بك\nHermes هو وكيل', [false, false])

    expect(hard.row(1)).toBe(`Hermes ${rev('هو وكيل')}`)
  })

  it('preserves cell widths, styles and the character multiset', () => {
    const line = 'مرحبا 你好 abc'
    const chars = [...line].map((value, i) => ({ hyperlink: undefined, styleId: i, value, width: stringWidth(value) }))
    const reordered = reorderBidi(chars)

    expect(reordered.map(c => c.value)).not.toEqual(chars.map(c => c.value))
    expect(reordered.reduce((n, c) => n + c.width, 0)).toBe(stringWidth(line))
    // Every cell moves with its own width and style.
    expect([...reordered].sort((a, b) => a.styleId - b.styleId)).toEqual(chars)

    // On screen the reordered line still ends at the same column.
    expect(stringWidth(paint(line).row(0))).toBe(stringWidth(line))
  })
})

describe('bidi reordering (software bidi off)', () => {
  beforeEach(() => setSoftwareBidiForTesting(false))
  afterEach(() => setSoftwareBidiForTesting(undefined))

  it('leaves RTL text in logical order for terminals that do bidi themselves', () => {
    const chars = clustered('العنوان 10.0.0.138')

    expect(reorderBidi(chars)).toBe(chars)
    expect(paint('العنوان 10.0.0.138').row(0)).toBe('العنوان 10.0.0.138')
    expect(paint('مرحبا بك\nHermes هو وكيل', [false, true]).row(1)).toBe('Hermes هو وكيل')
  })
})
