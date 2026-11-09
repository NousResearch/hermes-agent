/**
 * Bidirectional text reordering for terminal rendering.
 *
 * Most terminal emulators do NOT implement the Unicode Bidi Algorithm, so
 * RTL text (Hebrew, Arabic, etc.) is emitted in logical order and appears
 * reversed. This module applies the bidi algorithm to reorder ClusteredChar
 * arrays from logical order to visual order before Ink's LTR cell placement
 * loop.
 *
 * Whether reordering is needed is decided per terminal, not per platform.
 * A handful of terminals lay out RTL runs themselves — macOS CoreText
 * terminals, VTE-based terminals (GNOME Terminal, Tilix, Xfce Terminal…),
 * Konsole and mlterm — and on those software reordering would double-apply
 * and re-reverse the text. Everywhere else, including Windows Terminal and
 * conhost (https://github.com/microsoft/terminal/issues/538), WSL, VS Code's
 * xterm.js, Kitty, Alacritty, Ghostty, WezTerm (bidi shipped off) and plain
 * xterm, codepoints reach the screen in logical order with no bidi, so RTL
 * runs must be reordered here.
 *
 * Detection is overridable: HERMES_TUI_BIDI=1 forces software reordering on
 * (a terminal we misclassify, or a bidi-capable terminal the user wants
 * handled in-app); HERMES_TUI_BIDI=0 forces it off.
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

// Same override vocabulary as the rest of the TUI's env switches.
const ENV_ON_RE = /^(?:1|true|yes|on)$/i
const ENV_OFF_RE = /^(?:0|false|no|off)$/i

/**
 * Terminals that implement bidi layout themselves. On these we must leave
 * logical order untouched — reordering here would be applied twice.
 */
function terminalHasNativeBidi(env: NodeJS.ProcessEnv): boolean {
  const termProgram = (env.TERM_PROGRAM ?? '').trim()

  // macOS CoreText-backed terminals run the full bidi algorithm.
  if (termProgram === 'iTerm.app' || termProgram === 'Apple_Terminal') {
    return true
  }

  // VTE-based terminals do bidi themselves since VTE 0.58 (GNOME 3.34).
  // Same signal the TUI already uses for ANSI dim (shouldUseAnsiDim).
  if (env.VTE_VERSION) {
    return true
  }

  // Konsole lays out RTL runs through Qt's text engine.
  if (env.KONSOLE_VERSION) {
    return true
  }

  // mlterm handles bidi when built/configured with it.
  if (env.MLTERM || (env.TERM ?? '').trim() === 'mlterm') {
    return true
  }

  return false
}

/**
 * Whether Ink should reorder RTL runs itself before emitting to the terminal.
 * Almost every terminal lacks native bidi, so the default is on; HERMES_TUI_BIDI
 * overrides it in either direction.
 */
export function shouldReorderBidi(env: NodeJS.ProcessEnv = process.env): boolean {
  const override = (env.HERMES_TUI_BIDI ?? '').trim()

  if (ENV_ON_RE.test(override)) {
    return true
  }

  if (ENV_OFF_RE.test(override)) {
    return false
  }

  return !terminalHasNativeBidi(env)
}

function needsBidi(): boolean {
  if (needsSoftwareBidi === undefined) {
    needsSoftwareBidi = shouldReorderBidi()
  }

  return needsSoftwareBidi
}

/** Test-only: drop the cached decision so the next call re-detects. */
export function resetBidiDetectionForTest(): void {
  needsSoftwareBidi = undefined
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

  // Rule L4 — mirroring. Characters with the Bidi_Mirrored property that
  // resolve to an odd (RTL) level must be replaced by their mirror glyph: a
  // "(" laid out inside an RTL run reads as ")" and vice versa. A terminal
  // that runs the UBA does this itself; because we reorder in software we
  // must do it too, or a bracket pair displays glyph-swapped — e.g.
  // "שלום (עולם)" comes out ")םלוע( םולש" instead of "(םלוע) םולש".
  const mirrors = bidi.getMirroredCharactersMap(plainText, levels)

  // Map bidi levels back to ClusteredChar indices, applying the mirror at the
  // same time. Each ClusteredChar may be several code units in the joined
  // string, so the logical offset (UTF-16) is what indexes `levels`/`mirrors`.
  const charLevels: number[] = []
  const visual: ClusteredChar[] = []
  let offset = 0

  for (let i = 0; i < characters.length; i++) {
    const cluster = characters[i]!
    charLevels.push(levels[offset]!)
    const mirror = mirrors.get(offset)
    visual.push(mirror === undefined ? cluster : { ...cluster, value: mirror })
    offset += cluster.value.length
  }

  // Get reorder segments from bidi-js, but we need to work at the
  // ClusteredChar level, not the string level. We'll implement the
  // standard bidi reordering: find the max level, then for each level
  // from max down to 1, reverse all contiguous runs >= that level.
  const reordered = visual
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
