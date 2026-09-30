import { stringWidth } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

import { displayMathToUnicode } from '../lib/mathDisplay.js'
import { BOX_RE, texToUnicode } from '../lib/mathUnicode.js'

const matrix = (body: string, name = 'matrix') => `\\begin{${name}}${body}\\end{${name}}`

describe('displayMathToUnicode', () => {
  it('separates only top-level unescaped cells/rows, then converts and aligns scalar content', () => {
    for (const name of ['matrix', 'pmatrix', 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix']) {
      const body = String.raw`\alpha & \frac{1}{2} \\ x_1 & y^2`
      const lines = displayMathToUnicode(matrix(body, name)).split('\n')

      expect(lines).toHaveLength(2)
      expect(lines[0]).toContain('α   1/2')
      expect(lines[1]).toContain('x₁  y²')
      if (name !== 'matrix') {
        expect(stringWidth(lines[0]!)).toBe(stringWidth(lines[1]!))
      }
      expect(displayMathToUnicode(matrix(body + String.raw` \\ `, name))).toBe(lines.join('\n'))
      expect(displayMathToUnicode(matrix(body.replaceAll(' & ', '\n &\n '), name))).toBe(lines.join('\n'))
    }

    expect(displayMathToUnicode(matrix(String.raw`\text{a&b} & \{x\} \\ a\&b & \%`))).toBe('a&b  {x}\na&b  %')
    expect(displayMathToUnicode(matrix(String.raw`\{ & \} \\ c & d`))).toBe('{  }\nc  d')
    expect(displayMathToUnicode(matrix(String.raw`a & & c \\ d & e & f`))).toBe('a     c\nd  e  f')
    expect(displayMathToUnicode(matrix(String.raw`a \\\& b`))).toBe('a\n& b')
    expect(displayMathToUnicode(matrix('a\nb & c'))).toBe('a b  c')

    for (const body of [
      String.raw`\boxed{\alpha} & R \\ 12345 & S`,
      String.raw`\fbox{x_1} & R \\ abcdef & S`,
      String.raw`\text{中} & R \\ a & S`,
      String.raw`\hat{x} & R \\ abc & S`,
      String.raw`\mathbb{A} & R \\ a & S`
    ]) {
      const visible = displayMathToUnicode(matrix(body)).replace(BOX_RE, ' $1 ').split('\n')

      expect(visible).toHaveLength(2)
      expect(stringWidth(visible[0]!.slice(0, visible[0]!.indexOf('R')))).toBe(
        stringWidth(visible[1]!.slice(0, visible[1]!.indexOf('S')))
      )
    }

    const body = matrix(String.raw`\frac{1}{\frac{1}{x}} & \boxed{x^2}`)

    expect(displayMathToUnicode(body)).toBe(`1/(1/x)  ${texToUnicode(String.raw`\boxed{x^2}`)}`)
    expect(displayMathToUnicode(`A = ${matrix('a & b')} + \\alpha`)).toBe('A =\na  b\n+ α')
    const ordinary = ' \\alpha\n\n x_1 + \\boxed{y^2}'
    expect(displayMathToUnicode(ordinary)).toBe(ordinary.split('\n').map(texToUnicode).join('\n'))
  })

  it('falls back byte-for-byte on any unsupported or incomplete display environment', () => {
    const invalid = [
      String.raw`\begin`,
      String.raw`\begin{`,
      String.raw`\begin{matrix`,
      String.raw`\begin{matrix}1 & 2`,
      String.raw`\end{matrix}`,
      String.raw`\begin{matrix}1\end{pmatrix}`,
      String.raw`\begin{matrix}1\end`,
      String.raw`\begin{unknown}x\end{unknown}`,
      String.raw`\begin{matrix*}[r]1\end{matrix*}`,
      matrix(String.raw`\begin{matrix}1\end{matrix}`),
      matrix(String.raw`{a & b`),
      matrix(String.raw`a} & b`),
      matrix(String.raw`a & b \\ c`),
      matrix(''),
      matrix('  '),
      matrix(String.raw`a \\ \\`),
      matrix(String.raw`\\ a`),
      matrix(String.raw`a \\[2pt] b`),
      matrix(String.raw`a \\ [2pt] b`),
      matrix(String.raw`a \\* b`),
      matrix(String.raw`[r]a`),
      matrix(String.raw`a % comment`),
      matrix(String.raw`a \\ \hline b`),
      matrix(String.raw`\multicolumn{2}{c}{x}`),
      matrix(String.raw`a \cr b`),
      matrix(String.raw`\foo{x}`),
      matrix(String.raw`{a \\ b}`),
      matrix(String.raw`\boxed{\boxed{x}}`),
      matrix(String.raw`\@`),
      `{${matrix('a')}}`,
      `\\frac{${matrix('a')}}{2}`,
      `\\left(${matrix('a')}\\right)`,
      `${matrix('a')}^2`,
      `${matrix('a')} + ${matrix('b')}`,
      `${matrix('a')} + \\begin{unknown}b\\end{unknown}`,
      `${matrix('a')} + {`,
      `${matrix('a')} & b`,
      `${matrix('a')} \\\\ b`
    ]

    for (const body of invalid) {
      const display = `\\alpha + ${body}\n+ \\beta`

      expect(displayMathToUnicode(display), display).toBe(display)
    }

    // Escaped percent is not a comment, but an even number of backslashes
    // leaves percent unescaped after a row separator.
    const comment = matrix(String.raw`a \\% comment`)
    expect(displayMathToUnicode(comment)).toBe(comment)
  })
})
