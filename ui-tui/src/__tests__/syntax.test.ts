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
    expect(isHighlightable('csharp')).toBe(true)
    expect(isHighlightable('cs')).toBe(true)
    expect(isHighlightable('java')).toBe(true)
    expect(isHighlightable('kotlin')).toBe(true)
    expect(isHighlightable('whatever')).toBe(false)
    expect(isHighlightable('')).toBe(false)
  })

  it('paints a whole-line comment dim', () => {
    const tokens = highlightLine('// hello', 'ts', t)

    expect(tokens).toEqual([[t.color.muted, '// hello']])
  })

  it('paints keywords, strings, and numbers in a ts line', () => {
    const tokens = highlightLine(`const x = 'hi' + 42`, 'ts', t)
    const colors = tokens.map(tok => tok[0])

    expect(colors).toContain(t.color.border) // const
    expect(colors).toContain(t.color.accent) // 'hi'
    expect(colors).toContain(t.color.text) // 42
  })

  it('falls through unchanged for unknown langs', () => {
    const tokens = highlightLine(`const x = 1`, 'zzz', t)

    expect(tokens).toEqual([['', 'const x = 1']])
  })

  it('treats `#` as a python comment, not a selector', () => {
    const tokens = highlightLine('# comment', 'py', t)

    expect(tokens).toEqual([[t.color.muted, '# comment']])
  })

  it('highlights csharp aliases instead of falling back to plain text', () => {
    const tokens = highlightLine('public class Demo { }', 'csharp', t)
    const colors = tokens.map(tok => tok[0])

    expect(colors).toContain(t.color.border) // public / class
  })

  it('highlights java and kotlin fenced languages', () => {
    const javaTokens = highlightLine('public class Demo { }', 'java', t)
    const kotlinTokens = highlightLine('fun main() = true', 'kotlin', t)

    expect(javaTokens.map(tok => tok[0])).toContain(t.color.border) // public / class
    expect(kotlinTokens.map(tok => tok[0])).toContain(t.color.border) // fun / true
  })
})
