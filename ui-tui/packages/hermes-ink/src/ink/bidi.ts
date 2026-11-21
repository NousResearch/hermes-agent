/**
 * Bidirectional text reordering for terminal rendering.
 *
 * Some terminals do not implement the Unicode Bidirectional Algorithm (UAX #9),
 * so RTL text (Hebrew, Arabic, etc.) appears in logical order, i.e. reversed,
 * and digit runs inside RTL text come out backwards (10.0.0.138 -> 831.0.0.01).
 * This module reorders the ClusteredChar array of one already-wrapped line from
 * logical to visual order before Ink's LTR cell placement loop. Wrapping stays
 * logical; only the cells of each visual line are permuted, so every cell keeps
 * its width, style and hyperlink.
 *
 * Terminals that do bidi themselves (VTE/GNOME Terminal, Konsole, mlterm,
 * Terminal.app, iTerm2) must NOT get this, or the text is reordered twice.
 * Without bidi:
 * - Windows Terminal, including WSL (https://github.com/microsoft/terminal/issues/538),
 *   and conhost;
 * - xterm.js: the VS Code integrated terminal and the web dashboard's embedded
 *   chat (the dashboard sets HERMES_TUI_BIDI=1 on the PTY it spawns).
 *
 * HERMES_TUI_BIDI=1/0 (also true/false, yes/no, on/off) forces it on or off
 * and wins over the platform detection.
 */
import bidiFactory from 'bidi-js'

type ClusteredChar = {
  value: string
  width: number
  styleId: number
  hyperlink: string | undefined
}

/** Paragraph direction of a line, from its first strong character (UAX #9 P2/P3). */
export type BidiDirection = 'ltr' | 'rtl'

let bidiInstance: ReturnType<typeof bidiFactory> | undefined
let needsSoftwareBidi: boolean | undefined

function parseToggle(value: string | undefined): boolean | null {
  const raw = (value ?? '').trim()

  if (/^(?:1|true|yes|on)$/i.test(raw)) {
    return true
  }

  if (/^(?:0|false|no|off)$/i.test(raw)) {
    return false
  }

  return null
}

/**
 * Whether the TUI must reorder bidi text itself. HERMES_TUI_BIDI overrides;
 * otherwise only hosts known to lack native bidi get it.
 */
export function resolveSoftwareBidi(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): boolean {
  const override = parseToggle(env['HERMES_TUI_BIDI'])

  if (override !== null) {
    return override
  }

  return (
    platform === 'win32' ||
    typeof env['WT_SESSION'] === 'string' || // WSL in Windows Terminal
    env['TERM_PROGRAM'] === 'vscode' // VS Code integrated terminal (xterm.js)
  )
}

export function needsBidi(): boolean {
  if (needsSoftwareBidi === undefined) {
    needsSoftwareBidi = resolveSoftwareBidi()
  }

  return needsSoftwareBidi
}

/** Test hook: force the gate (true/false) or re-read the environment (undefined). */
export function setSoftwareBidiForTesting(value: boolean | undefined): void {
  needsSoftwareBidi = value
}

function getBidi() {
  if (!bidiInstance) {
    bidiInstance = bidiFactory()
  }

  return bidiInstance
}

/**
 * Direction of the first strong character in `text` (plain text, no ANSI), or
 * undefined when it has none. Used to carry a paragraph's direction onto its
 * soft-wrapped continuation lines, which UAX #9 treats as one paragraph.
 */
export function firstStrongDirection(text: string): BidiDirection | undefined {
  const bidi = getBidi()

  for (const ch of text) {
    const type = bidi.getBidiCharTypeName(ch)

    if (type === 'L') {
      return 'ltr'
    }

    if (type === 'R' || type === 'AL') {
      return 'rtl'
    }
  }

  return undefined
}

/**
 * Reorder an array of ClusteredChars (one visual line) from logical order to
 * visual order using the Unicode Bidi Algorithm, mirroring paired glyphs such
 * as brackets in RTL runs (rule L4). `direction` is the paragraph direction;
 * undefined resolves it from this line's first strong character.
 *
 * Returns the same array when the gate is off or the line has no RTL text.
 */
export function reorderBidi(characters: ClusteredChar[], direction?: BidiDirection): ClusteredChar[] {
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
  const { levels } = bidi.getEmbeddingLevels(plainText, direction ?? 'auto')

  // Map bidi levels back to ClusteredChar indices.
  // Each ClusteredChar may be multiple code units in the joined string.
  const charLevels: number[] = []
  const reordered: ClusteredChar[] = []
  let offset = 0

  for (let i = 0; i < characters.length; i++) {
    const character = characters[i]!
    const level = levels[offset]!
    charLevels.push(level)

    // L4: a mirrored character ('(', '[', '<', '«', ...) at an odd (RTL)
    // level is shown as its mirror image. Same width, so cells are unchanged.
    const mirror = (level & 1) === 1 && character.value.length === 1 ? bidi.getMirroredCharacter(character.value) : null
    reordered.push(mirror ? { ...character, value: mirror } : character)
    offset += character.value.length
  }

  // L2: from the highest level down to the lowest odd level, reverse every
  // contiguous run at that level or higher.
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
        reverseRange(charLevels, i, j - 1)
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

/**
 * Quick check for RTL characters (Hebrew, Arabic, and related scripts).
 * Avoids running the full bidi algorithm on pure-LTR text.
 */
export function hasRTLCharacters(text: string): boolean {
  // Hebrew: U+0590-U+05FF, U+FB1D-U+FB4F
  // Arabic: U+0600-U+06FF, U+0750-U+077F, U+08A0-U+08FF, U+FB50-U+FDFF, U+FE70-U+FEFF
  // Thaana: U+0780-U+07BF
  // Syriac: U+0700-U+074F
  return /[\u0590-\u05FF\uFB1D-\uFB4F\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0780-\u07BF\u0700-\u074F]/u.test(
    text
  )
}
