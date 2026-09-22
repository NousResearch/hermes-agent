/**
 * Bidirectional text reordering for terminal rendering.
 *
 * Terminals on Windows do not implement the Unicode Bidi Algorithm,
 * so RTL text (Hebrew, Arabic, etc.) appears reversed. This module
 * applies the bidi algorithm to reorder ClusteredChar arrays from
 * logical order to visual order before Ink's LTR cell placement loop.
 *
 * On macOS terminals (Terminal.app, iTerm2) bidi works natively.
 * Windows Terminal (including WSL) does not implement bidi
 * (https://github.com/microsoft/terminal/issues/538).
 *
 * Detection: Windows Terminal sets WT_SESSION; native Windows cmd/conhost
 * also lacks bidi. We enable bidi reordering when running on Windows or
 * inside Windows Terminal (covers WSL).
 */
import bidiFactory from 'bidi-js'
import stripAnsi from 'strip-ansi'

import { shapeArabicCharacters } from './arabic.js'
import { isDashboardHosted } from './termio/host.js'

type ClusteredChar = {
  value: string
  width: number
  styleId: number
  hyperlink: string | undefined
}

let bidiInstance: ReturnType<typeof bidiFactory> | undefined
let needsSoftwareBidi: boolean | undefined

export function setNeedsBidiForTesting(val: boolean | undefined): void {
  needsSoftwareBidi = val
}

function needsBidi(): boolean {
  if (needsSoftwareBidi === undefined) {
    needsSoftwareBidi =
      process.env['HERMES_ENABLE_BIDI'] === '1' ||
      isDashboardHosted() ||
      process.platform === 'win32' ||
      process.platform === 'linux' ||
      typeof process.env['WT_SESSION'] === 'string' || // WSL in Windows Terminal
      process.env['TERM_PROGRAM'] === 'vscode' // VS Code integrated terminal (xterm.js)
  }

  return needsSoftwareBidi
}

function getBidi() {
  if (!bidiInstance) {
    bidiInstance = bidiFactory()
  }

  return bidiInstance
}

/**
 * Mirror brackets when reversed within an RTL run (odd bidi embedding level).
 */
function mirrorBidiBracket(char: string): string {
  switch (char) {
    case '(':
      return ')'

    case ')':
      return '('

    case '[':
      return ']'

    case ']':
      return '['

    case '{':
      return '}'

    case '}':
      return '{'

    case '<':
      return '>'

    case '>':
      return '<'

    case '«':
      return '»'

    case '»':
      return '«'

    default:
      return char
  }
}

/**
 * Determine the base paragraph direction for a line.
 * Lines beginning with code syntax, tool names (e.g. `Terminal(...)`), command prompts,
 * or Latin identifiers get an LTR base direction to protect code syntax and parentheses
 * from mirroring or scrambling. Conversational text starting with RTL characters gets RTL.
 */
export function getParagraphDirection(text: string): 'ltr' | 'rtl' {
  // Strip ANSI escape sequences
  const clean = stripAnsi(text).trim()

  for (let i = 0; i < clean.length; i++) {
    const code = clean.codePointAt(i)

    if (!code) {
      continue
    }

    // Skip bullet points, markdown list markers, or leading symbols
    if (code === 0x2022 || code === 0x2d || code === 0x2a || code === 0x23 || code <= 0x20) {
      continue
    }

    // RTL: Arabic, Hebrew, Syriac, Thaana
    if (
      (code >= 0x0590 && code <= 0x08ff) ||
      (code >= 0xfb1d && code <= 0xfdff) ||
      (code >= 0xfe70 && code <= 0xfeff)
    ) {
      return 'rtl'
    }

    // LTR: Latin, ASCII digits, CJK, Cyrillic, Greek, or code prefixes ('/', '$', '>')
    if (
      (code >= 0x0041 && code <= 0x005a) ||
      (code >= 0x0061 && code <= 0x007a) ||
      (code >= 0x0030 && code <= 0x0039) ||
      (code >= 0x00c0 && code <= 0x024f) ||
      (code >= 0x0400 && code <= 0x04ff) ||
      (code >= 0x4e00 && code <= 0x9fff) ||
      code === 0x2f ||
      code === 0x24 ||
      code === 0x3e
    ) {
      return 'ltr'
    }
  }

  return 'ltr'
}

/**
 * Reorder an array of ClusteredChars from logical order to visual order
 * using the Unicode Bidi Algorithm with Arabic contextual shaping.
 * Active on terminals that lack native bidi/shaping support (Web Dashboard,
 * Windows Terminal, conhost, WSL, Linux consoles).
 *
 * Returns the same array on bidi-capable native terminals (no-op).
 */
export function reorderBidi(characters: ClusteredChar[]): ClusteredChar[] {
  if (!needsBidi() || characters.length === 0) {
    return characters
  }

  // Build a plain string from the clustered chars to check for RTL scripts
  const plainText = characters.map(c => c.value).join('')

  if (!hasRTLCharacters(plainText)) {
    return characters
  }

  // 1. Shape Arabic characters in logical order before visual reordering
  const shaped = shapeArabicCharacters(characters)
  const shapedText = shaped.map(c => c.value).join('')

  // 2. Compute bidi embedding levels with context-aware paragraph direction
  const bidi = getBidi()
  const dir = getParagraphDirection(shapedText)
  const { levels } = bidi.getEmbeddingLevels(shapedText, dir)

  // Map bidi levels back to ClusteredChar indices.
  // Each ClusteredChar may be multiple code units in the joined string.
  const charLevels: number[] = []
  let offset = 0

  for (let i = 0; i < shaped.length; i++) {
    charLevels.push(levels[offset]!)
    offset += shaped[i]!.value.length
  }

  // 3. Reorder runs by bidi level (from max level down to 1)
  const reordered = [...shaped]
  const maxLevel = Math.max(...charLevels)

  for (let level = maxLevel; level >= 1; level--) {
    let i = 0

    while (i < reordered.length) {
      if (charLevels[i]! >= level) {
        let j = i + 1

        while (j < reordered.length && charLevels[j]! >= level) {
          j++
        }

        // Reverse the run
        reverseRange(reordered, i, j - 1)
        reverseRangeNumbers(charLevels, i, j - 1)

        // If this level is odd (RTL), mirror paired brackets
        if (level % 2 === 1) {
          for (let k = i; k < j; k++) {
            const item = reordered[k]!
            const mirrored = mirrorBidiBracket(item.value)

            if (mirrored !== item.value) {
              reordered[k] = { ...item, value: mirrored }
            }
          }
        }

        i = j
      } else {
        i++
      }
    }
  }

  return reordered
}

function reverseRange<T>(arr: T[], start: number, end: number): void {
  while (start < end) {
    const temp = arr[start]!
    arr[start] = arr[end]!
    arr[end] = temp
    start++
    end--
  }
}

function reverseRangeNumbers(arr: number[], start: number, end: number): void {
  while (start < end) {
    const temp = arr[start]!
    arr[start] = arr[end]!
    arr[end] = temp
    start++
    end--
  }
}

/**
 * Quick check for RTL characters (Hebrew, Arabic, and related scripts).
 * Avoids running the full bidi algorithm on pure-LTR text.
 */
function hasRTLCharacters(text: string): boolean {
  // Hebrew: U+0590-U+05FF, U+FB1D-U+FB4F
  // Arabic: U+0600-U+06FF, U+0750-U+077F, U+08A0-U+08FF, U+FB50-U+FDFF, U+FE70-U+FEFF
  // Thaana: U+0780-U+07BF
  // Syriac: U+0700-U+074F
  return /[\u0590-\u05FF\uFB1D-\uFB4F\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0780-\u07BF\u0700-\u074F]/u.test(
    text
  )
}
