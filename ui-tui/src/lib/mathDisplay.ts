import { stringWidth } from '@hermes/ink'

import { BOX_CLOSE, BOX_OPEN, BOX_RE, texToUnicode } from './mathUnicode.js'

const DELIMITERS: Record<string, readonly [string, string]> = {
  matrix: ['', ''],
  pmatrix: ['( ', ' )'],
  bmatrix: ['[ ', ' ]'],
  Bmatrix: ['{ ', ' }'],
  vmatrix: ['| ', ' |'],
  Vmatrix: ['|| ', ' ||']
}

// Only flat, standalone environments (with optional scalar text before/after).
// Validate the whole display BEFORE any scalar replacements: a broken streaming
// fragment must not turn into a mixture of rendered cells and raw TeX.
export function displayMathToUnicode(input: string): string {
  if (!/\\(?:begin|end)(?![A-Za-z])/.test(input)) {
    return input.split('\n').map(texToUnicode).join('\n')
  }

  let depth = 0
  let name: string | undefined
  let start = -1
  let end = -1
  let cellStart = 0
  let row: string[] = []
  const rows: string[][] = []

  for (let i = 0; i < input.length; i++) {
    const ch = input[i]!

    if (ch === '%' || ch === BOX_OPEN || ch === BOX_CLOSE) {
      return input
    }

    if (ch === '{') {
      depth++
    }

    if (ch === '}' && --depth < 0) {
      return input
    }

    if (ch === '&' && depth === 0) {
      if (!name) {
        return input
      }

      row.push(input.slice(cellStart, i))
      cellStart = i + 1
    }

    if (ch !== '\\') {
      continue
    }

    const command = /^\\([A-Za-z]+|[^\r\n])/.exec(input.slice(i))

    if (!command) {
      return input
    }

    const word = command[1]!

    if (word === 'begin' || word === 'end') {
      const env = /^\\(?:begin|end)\s*\{([A-Za-z]+)\}/.exec(input.slice(i))

      if (!env || depth !== 0) {
        return input
      }

      if (word === 'begin') {
        if (start !== -1 || !Object.hasOwn(DELIMITERS, env[1]!)) {
          return input
        }

        name = env[1]!
        start = i
        cellStart = i + env[0].length

        // Optional alignment arguments are not part of the flat matrix subset.
        if (/^\s*\[/.test(input.slice(cellStart))) {
          return input
        }
      } else {
        if (!name || env[1] !== name) {
          return input
        }

        const last = input.slice(cellStart, i)

        if (last.trim() || row.length) {
          rows.push([...row, last])
        }

        // A single trailing row separator is allowed, not empty matrices/rows.
        if (!rows.length) {
          return input
        }

        end = i + env[0].length
        name = undefined
      }

      i += env[0].length - 1
    } else if (word === '\\') {
      if (!name || depth !== 0 || /^\s*[\[*]/.test(input.slice(i + 2))) {
        return input
      }

      row.push(input.slice(cellStart, i))

      if (row.length === 1 && !row[0]!.trim()) {
        return input
      }

      rows.push(row)
      row = []
      cellStart = i + 2
      i++
    } else {
      // Escaped braces and ampersands are literals, never structural tokens.
      i += command[0].length - 1
    }
  }

  if (depth !== 0 || name || start < 0 || end < 0) {
    return input
  }

  if (rows.some(r => r.length !== rows[0]!.length)) {
    return input
  }

  const before = input.slice(0, start)
  const after = input.slice(end)

  // A delimiter wrapper around a grid needs baseline layout, not scalar lines.
  if (/\\(?:left|right)(?![A-Za-z])/.test(before + after) || /^\s*[_^]/.test(after)) {
    return input
  }

  const scalar = (text: string) => texToUnicode(text.replace(/\s+/g, ' ').trim())
  const prefix = scalar(before)
  const suffix = scalar(after)
  const cells = rows.map(r => r.map(scalar))
  // Unsupported commands (row rules, spans, macros, etc.) stay raw atomically.
  // Nested box sentinels cannot be measured like renderMath's single highlight.
  const converted = [prefix, suffix, ...cells.flat()]

  if (
    converted.some(
      s => s.includes('\\') || s.replace(BOX_RE, '').includes(BOX_OPEN) || s.replace(BOX_RE, '').includes(BOX_CLOSE)
    )
  ) {
    return input
  }

  const width = (s: string) => stringWidth(s.replace(BOX_RE, ' $1 '))
  const widths = cells[0]!.map((_, col) => Math.max(...cells.map(r => width(r[col]!))))
  const envName = /^\\begin\s*\{([A-Za-z]+)\}/.exec(input.slice(start))![1]!
  const [left, right] = DELIMITERS[envName]!
  const grid = cells.map(
    r =>
      left +
      r
        .map((cell, col) => {
          const gap = col < r.length - 1 ? 2 : 0
          const padding = gap || right ? widths[col]! - width(cell) + gap : 0

          return cell + ' '.repeat(padding)
        })
        .join('') +
      right
  )

  // Keep surrounding scalar expressions in reading order, on separate lines;
  // do not pretend to implement general TeX baseline or multi-matrix layout.
  return [...(prefix ? [prefix] : []), ...grid, ...(suffix ? [suffix] : [])].join('\n')
}
