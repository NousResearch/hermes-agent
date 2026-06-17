import { describe, expect, it } from 'vitest'

import { highlightLine, isHighlightable } from '../lib/syntax.js'
import { DEFAULT_THEME } from '../theme.js'

const t = DEFAULT_THEME

describe('syntax highlighter', () => {
  it('recognizes supported langs and aliases', () => {
    expect(isHighlightable('ts')).toBe(true)
    expect(isHighlightable('js')).toBe(true)
    expect(isHighlightable('python')).toBe(true)
    expect(isHighlightable('rs')).toBe(true)
    expect(isHighlightable('bash')).toBe(true)
    expect(isHighlightable('whatever')).toBe(false)
    expect(isHighlightable('')).toBe(false)
  })

  it('paints a whole-line comment dim', () => {
    const tokens = highlightLine('// hello', 'ts', t)

    expect(tokens).toEqual([[t.color.syntaxComment, '// hello']])
  })

  it('paints keywords, strings, numbers, functions, and operators in a ts line', () => {
    const tokens = highlightLine(`const x = fn('hi') + 42`, 'ts', t)
    const colors = tokens.map(tok => tok[0])

    expect(colors).toContain(t.color.syntaxKeyword) // const
    expect(colors).toContain(t.color.syntaxFunction) // fn
    expect(colors).toContain(t.color.syntaxString) // 'hi'
    expect(colors).toContain(t.color.syntaxOperator) // +
    expect(colors).toContain(t.color.syntaxNumber) // 42
  })

  it('falls through unchanged for unknown langs', () => {
    const tokens = highlightLine(`const x = 1`, 'zzz', t)

    expect(tokens).toEqual([['', 'const x = 1']])
  })

  it('treats `#` as a python comment, not a selector', () => {
    const tokens = highlightLine('# comment', 'py', t)

    expect(tokens).toEqual([[t.color.syntaxComment, '# comment']])
  })
})
