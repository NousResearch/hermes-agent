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
 * A handful of terminals lay out RTL runs themselves — macOS Terminal.app,
 * VTE-based terminals (GNOME Terminal, Tilix, Xfce Terminal…) from 0.58,
 * Konsole from 22.12, and mlterm — and on those software reordering would
 * double-apply and re-reverse the text. Everywhere else, including Windows
 * Terminal and conhost (https://github.com/microsoft/terminal/issues/538),
 * WSL, VS Code's xterm.js, Kitty, Alacritty, Ghostty, plain xterm, and the
 * two that have bidi but don't use it here (WezTerm ships it off, iTerm2
 * hides it behind an advanced setting and skips the alternate screen),
 * codepoints reach the screen in logical order with no bidi, so RTL runs
 * must be reordered here.
 *
 * Detection is overridable: HERMES_TUI_BIDI=1 forces software reordering on
 * (e.g. a terminal we misclassify, or a bidi-capable terminal the user wants
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

/** VTE and Konsole both export their version packed as MAJOR*10000 +
 *  MINOR*100 + PATCH (e.g. 7603 = 0.76.3, 240800 = 24.8.0). Anything
 *  unparseable counts as unknown, and unknown means we reorder. */
function packedVersion(raw: string | undefined): number | undefined {
  const version = parseInt((raw ?? '').trim(), 10)

  return Number.isFinite(version) && version > 0 ? version : undefined
}

/** VTE gained bidi in 0.58 (GNOME 3.34). */
const VTE_BIDI_MIN = 5800

/** Konsole reorders through ICU from 22.12. Earlier builds ran the bidi
 *  algorithm per run of identical attributes, so a coloured line came out
 *  wrong (invent.kde.org MR 722, KDE bug 403729). */
const KONSOLE_BIDI_MIN = 221200

/**
 * Terminals that implement bidi layout themselves. On these we must leave
 * logical order untouched — reordering here would be applied twice.
 */
function terminalHasNativeBidi(env: NodeJS.ProcessEnv): boolean {
  const termProgram = (env.TERM_PROGRAM ?? '').trim()

  // macOS CoreText-backed Terminal.app runs the full bidi algorithm.
  // iTerm2 deliberately isn't here: its bidi is opt-in (Settings > Advanced >
  // Support right-to-left-text) and doesn't cover the alternate screen, which
  // is exactly where fullscreen mode lives. Same reasoning as WezTerm below.
  if (termProgram === 'Apple_Terminal') {
    return true
  }

  // VTE-based terminals (GNOME Terminal, Tilix, Xfce Terminal, Guake…). The
  // version gate matters twice: bidi arrived in 0.58, and fribidi is an
  // optional build dependency, so a terminal can report a version and still
  // lay RTL out in logical order. shouldUseAnsiDim reads the same variable.
  const vteVersion = packedVersion(env.VTE_VERSION)

  if (vteVersion !== undefined) {
    return vteVersion >= VTE_BIDI_MIN
  }

  // Konsole lays out RTL runs through ICU from 22.12.
  const konsoleVersion = packedVersion(env.KONSOLE_VERSION)

  if (konsoleVersion !== undefined) {
    return konsoleVersion >= KONSOLE_BIDI_MIN
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

export type BidiReorderResult = {
  /** Characters in visual order, ready for LTR cell placement. */
  visual: ClusteredChar[]
  /**
   * Index map from visual position to LOGICAL position: `visual[i]` is the
   * input character at `perm[i]` (modulo L4 mirroring, which is a glyph
   * substitution applied at the same position). `undefined` when nothing was
   * reordered — then `visual` IS the input array.
   *
   * The screen buffer only ever holds VISUAL order (that is what the terminal
   * paints), so text selection must invert this map to hand back logical
   * order. Without it, copying an RTL line out of the TUI yields the reversed
   * run and pastes unreadable.
   */
  perm: number[] | undefined
}

/**
 * Reorder an array of ClusteredChars from logical order to visual order
 * using the Unicode Bidi Algorithm, also returning the position map needed
 * to turn a visual column range back into logical order (see BidiReorderResult).
 *
 * Active on terminals that lack native bidi support (everything except macOS
 * CoreText terminals, VTE-based terminals, Konsole and mlterm).
 *
 * Base direction stays UAX #9 'auto' (first-strong): a line that starts with
 * neutral chrome and then RTL gets an RTL base and takes its indent with it.
 * That's a policy question, tracked separately — this only hands back the map
 * that turns visual columns into logical order.
 *
 * Returns the same array on bidi-capable terminals (no-op).
 */
export function reorderBidiWithMapping(characters: ClusteredChar[]): BidiReorderResult {
  const identity: BidiReorderResult = { visual: characters, perm: undefined }

  if (!needsBidi() || characters.length === 0) {
    return identity
  }

  // Build a plain string from the clustered chars to run through bidi
  const plainText = characters.map(c => c.value).join('')

  // Check if there are any RTL characters — skip bidi if pure LTR
  if (!hasRTLCharacters(plainText)) {
    return identity
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

  // Position map, permuted by exactly the same reversals so that after the
  // loop `order[i]` is the logical index of the character now at visual
  // position i. Selection copy inverts it (see BidiReorderResult.perm).
  const order: number[] = visual.map((_, i) => i)

  for (let level = maxLevel; level >= 1; level--) {
    let i = 0

    while (i < reordered.length) {
      if (charLevels[i]! >= level) {
        // Find the end of this run
        let j = i + 1

        while (j < reordered.length && charLevels[j]! >= level) {
          j++
        }

        // Reverse the run in all three arrays — keeping `order` in lockstep
        // is what makes the map exact (no re-derivation at copy time).
        reverseRange(reordered, i, j - 1)
        reverseRangeNumbers(charLevels, i, j - 1)
        reverseRangeNumbers(order, i, j - 1)
        i = j
      } else {
        i++
      }
    }
  }

  return { visual: reordered, perm: order }
}

/**
 * Visual-order only wrapper around `reorderBidiWithMapping` — the render hot
 * path only needs the characters. Returns the input array unchanged when
 * reordering is off or the line is pure LTR.
 */
export function reorderBidi(characters: ClusteredChar[]): ClusteredChar[] {
  return reorderBidiWithMapping(characters).visual
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
