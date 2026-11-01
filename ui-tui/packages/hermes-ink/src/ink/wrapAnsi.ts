import wrapAnsiNpm from 'wrap-ansi'

import { stringWidth } from './stringWidth.js'

type WrapAnsiOptions = {
  hard?: boolean
  wordWrap?: boolean
  trim?: boolean
}

const wrapAnsiBun = typeof Bun !== 'undefined' && typeof Bun.wrapAnsi === 'function' ? Bun.wrapAnsi : null

const wrapAnsiBackend: (input: string, columns: number, options?: WrapAnsiOptions) => string =
  wrapAnsiBun ?? wrapAnsiNpm

// Scripts that wrap between characters instead of at spaces (Han, Hiragana,
// Katakana, Hangul). wrap-ansi splits on ' ' only, so a whole CJK clause is
// one unbreakable "word": when the next clause does not fit the remainder of
// the row it moves to the next line whole, leaving most of the previous row
// blank (a 120-col transcript can end a line at col 60). Treat each
// ideograph as its own break opportunity instead.
const IDEOGRAPH_RE = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u

// Closing/trailing punctuation may not start a row in Chinese/Japanese
// typography; pull the preceding ideograph down with it when a break lands
// right before one of these. One character class, three readers: the wrapper's
// break rule, the "is this token only punctuation" merge test, and the
// leading-punctuation split in `unbreakableUnits`.
const CLOSING_PUNCTUATION = '、。，．,.;:!?！？：；）)\\]】」』》〉”’…—～'
const NO_LINE_START_RE = new RegExp(`^[${CLOSING_PUNCTUATION}]`, 'u')
const TRAILING_PUNCTUATION_RE = new RegExp(`^[${CLOSING_PUNCTUATION}]+$`, 'u')
const LEADING_PUNCTUATION_RE = new RegExp(`^[${CLOSING_PUNCTUATION}]+`, 'u')

// ANSI escapes have no layout width and must not be split; strings that
// carry them keep the stock backend.
const hasEscape = (text: string): boolean => text.includes('\u001B')

const SEGMENTER =
  typeof Intl !== 'undefined' && 'Segmenter' in Intl
    ? new (
        Intl as { Segmenter: new (l?: string, o?: object) => { segment: (s: string) => Iterable<{ segment: string }> } }
      ).Segmenter(undefined, {
        granularity: 'grapheme'
      })
    : null

const graphemes = (text: string): string[] =>
  SEGMENTER ? [...SEGMENTER.segment(text)].map(part => part.segment) : [...text]

const isIdeograph = (grapheme: string): boolean => IDEOGRAPH_RE.test(grapheme)

const displayWidth = (text: string): number => stringWidth(text)

type Token = { kind: 'char' | 'space' | 'word'; text: string; width: number }

// Split one source line into wrap tokens: whitespace, single ideographs
// (break opportunities), and runs of everything else (Latin words, digits,
// ASCII punctuation) that must stay unbroken.
const tokenize = (line: string): Token[] => {
  const tokens: Token[] = []
  let word = ''

  const flushWord = () => {
    if (word) {
      tokens.push({ kind: 'word', text: word, width: displayWidth(word) })
      word = ''
    }
  }

  for (const grapheme of graphemes(line)) {
    if (/\s/.test(grapheme)) {
      flushWord()
      tokens.push({ kind: 'space', text: grapheme, width: 1 })
    } else if (isIdeograph(grapheme)) {
      flushWord()
      tokens.push({ kind: 'char', text: grapheme, width: displayWidth(grapheme) })
    } else {
      word += grapheme
    }
  }

  flushWord()

  return tokens
}

const trimEnd = (line: string): string => line.replace(/\s+$/u, '')

const cjkWrapLine = (line: string, columns: number, options: WrapAnsiOptions): string[] => {
  const trim = options.trim !== false
  const hard = options.hard !== false
  const rows: string[] = []
  let row = ''
  let width = 0

  const pushRow = () => {
    rows.push(trim ? trimEnd(row) : row)
    row = ''
    width = 0
  }

  const appendHardBroken = (token: Token) => {
    // A single token wider than the row: fill it grapheme by grapheme.
    for (const grapheme of graphemes(token.text)) {
      const gw = displayWidth(grapheme)

      if (width > 0 && width + gw > columns) {
        pushRow()
      }

      row += grapheme
      width += gw
    }
  }

  for (const token of tokenize(line)) {
    if (token.kind === 'space') {
      if (width === 0) {
        // Leading indentation survives only when trimming is off, matching
        // wrap-ansi's trim:false behavior.
        if (!trim) {
          row += token.text
          width += token.width
        }

        continue
      }

      // wrap-ansi appends the separating space to the row *before* deciding
      // the next word needs its own row; keeping that space in the output is
      // what lets the caller map wrapped rows back to source characters.
      if (!trim || width + token.width <= columns) {
        row += token.text
        width += token.width
      }

      continue
    }

    if (width + token.width <= columns) {
      row += token.text
      width += token.width

      continue
    }

    if (width === 0) {
      if (hard && token.width > columns) {
        appendHardBroken(token)
      } else {
        row = token.text
        width = token.width
      }

      continue
    }

    // Keep CJK closing punctuation off the start of the new row by dragging
    // the preceding ideograph down with it (a space at the break is dropped,
    // matching word wrap).
    if (NO_LINE_START_RE.test(token.text)) {
      const chars = graphemes(row.replace(/\s+$/u, ''))
      const carry = chars.pop() ?? ''

      if (carry && isIdeograph(carry)) {
        row = chars.join('')
        pushRow()
        row = carry + token.text
        width = displayWidth(row)

        continue
      }
    }

    pushRow()

    if (hard && token.width > columns) {
      appendHardBroken(token)
    } else {
      row = token.text
      width = token.width
    }
  }

  if (row !== '' || rows.length === 0) {
    rows.push(trim ? trimEnd(row) : row)
  }

  return rows
}

/**
 * The pieces of `text` that a wrapper may not split: Latin words, and single
 * ideographs (with any closing punctuation the wrapper refuses to start a row
 * with, since it drags the ideograph down with it). Callers that size columns
 * or cells — markdown tables — use this as the minimum width a column needs.
 */
const unbreakableUnits = (text: string): string[] => {
  const units: string[] = []
  let word = ''

  const flush = () => {
    if (word) {
      units.push(word)
      word = ''
    }
  }

  for (const grapheme of graphemes(text)) {
    if (/\s/.test(grapheme)) {
      flush()
    } else if (isIdeograph(grapheme)) {
      flush()
      units.push(grapheme)
    } else {
      word += grapheme
    }
  }

  flush()

  const merged: string[] = []

  for (const unit of units) {
    const previous = merged.length - 1
    // A punctuation-led token ("，abc") belongs partly to the ideograph before
    // it: pull the punctuation run off the front and glue it to that
    // ideograph, keeping any Latin tail as its own unit.
    const leadingPunctuation = TRAILING_PUNCTUATION_RE.test(unit) ? unit : LEADING_PUNCTUATION_RE.exec(unit)?.[0]

    if (previous >= 0 && leadingPunctuation && IDEOGRAPH_RE.test(merged[previous]!)) {
      merged[previous] += leadingPunctuation

      const rest = unit.slice(leadingPunctuation.length)

      if (rest) {
        merged.push(rest)
      }

      continue
    }

    merged.push(unit)
  }

  return merged
}

/**
 * Wrap a string to `columns` display cells. Latin text keeps the stock
 * wrap-ansi word wrap; text containing CJK ideographs uses a per-character
 * aware wrap so rows fill instead of ending at the previous space. ANSI
 * escape sequences always take the stock path (they carry no layout width).
 */
const wrapAnsi = (input: string, columns: number, options: WrapAnsiOptions = {}): string => {
  const text = String(input)

  if (columns <= 0 || options.wordWrap === false || hasEscape(text) || !IDEOGRAPH_RE.test(text)) {
    return wrapAnsiBackend(text, columns, options)
  }

  return text
    .replaceAll('\r\n', '\n')
    .split('\n')
    .map(line => cjkWrapLine(line.normalize(), columns, options).join('\n'))
    .join('\n')
}

export { unbreakableUnits, wrapAnsi }
