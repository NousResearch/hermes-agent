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

type ClusteredChar = {
  value: string
  width: number
  styleId: number
  hyperlink: string | undefined
}

let bidiInstance: ReturnType<typeof bidiFactory> | undefined
let needsSoftwareBidi: boolean | undefined

function needsBidi(): boolean {
  if (needsSoftwareBidi === undefined) {
    needsSoftwareBidi =
      process.platform === 'win32' ||
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
 * Reorder an array of ClusteredChars from logical order to visual order
 * using the Unicode Bidi Algorithm. Active on terminals that lack native
 * bidi support (Windows Terminal, conhost, WSL).
 *
 * Returns the same array on bidi-capable terminals (no-op).
 */
export function reorderBidi(characters: ClusteredChar[]): ClusteredChar[] {
  if (!needsBidi() || characters.length === 0) {
    return characters
  }

  // Build a plain string from the clustered chars to run through bidi
  const plainText = characters.map(c => c.value).join('')

  // Check if there are any RTL characters — skip bidi if pure LTR
  if (!hasRTLCharacters(plainText)) {
    return characters
  }

  const bidi = getBidi()
  const { levels } = bidi.getEmbeddingLevels(plainText, 'auto')

  // Map bidi levels back to ClusteredChar indices.
  // Each ClusteredChar may be multiple code units in the joined string.
  const charLevels: number[] = []
  let offset = 0

  for (let i = 0; i < characters.length; i++) {
    charLevels.push(levels[offset]!)
    offset += characters[i]!.value.length
  }

  // Get reorder segments from bidi-js, but we need to work at the
  // ClusteredChar level, not the string level. We'll implement the
  // standard bidi reordering: find the max level, then for each level
  // from max down to 1, reverse all contiguous runs >= that level.
  const reordered = [...characters]
  const maxLevel = Math.max(...charLevels)

  for (let level = maxLevel; level >= 1; level--) {
    let i = 0

    while (i < reordered.length) {
      if (charLevels[i]! >= level) {
        // Find the end of this run
        let j = i + 1

        while (j < reordered.length && charLevels[j]! >= level) {
          j++
        }

        // Reverse the run in both arrays
        reverseRange(reordered, i, j - 1)
        reverseRangeNumbers(charLevels, i, j - 1)
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
