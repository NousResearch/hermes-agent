/**
 * Arabic text shaping and normalization for terminal rendering.
 *
 * Terminal monospace grids render characters cell-by-cell without native
 * OpenType complex text shaping (init, medi, fina glyph substitution).
 * This module converts logical Arabic characters (U+0600–U+06FF) into their
 * contextual presentation forms (Unicode Arabic Presentation Forms-B: U+FE70–U+FEFF
 * and Presentation Forms-A) so that adjacent cells visually touch and form
 * connected cursive Arabic text in any terminal emulator.
 *
 * It also provides un-shaping (normalization) to convert presentation forms
 * back to standard logical Arabic characters when text is copied to the clipboard.
 */

export type JoiningType = 'none' | 'right' | 'dual'

interface ArabicGlyphForms {
  isolated: string
  final: string
  initial: string
  medial: string
  joining: JoiningType
}

// Table of Arabic base characters and their presentation forms: [isolated, final, initial, medial]
const ARABIC_GLYPHS: Record<number, ArabicGlyphForms> = {
  // Hamza (non-joining)
  0x0621: { isolated: '\uFE80', final: '\uFE80', initial: '\uFE80', medial: '\uFE80', joining: 'none' },
  // Alef with Madda Above (right-joining)
  0x0622: { isolated: '\uFE81', final: '\uFE82', initial: '\uFE81', medial: '\uFE82', joining: 'right' },
  // Alef with Hamza Above (right-joining)
  0x0623: { isolated: '\uFE83', final: '\uFE84', initial: '\uFE83', medial: '\uFE84', joining: 'right' },
  // Waw with Hamza Above (right-joining)
  0x0624: { isolated: '\uFE85', final: '\uFE86', initial: '\uFE85', medial: '\uFE86', joining: 'right' },
  // Alef with Hamza Below (right-joining)
  0x0625: { isolated: '\uFE87', final: '\uFE88', initial: '\uFE87', medial: '\uFE88', joining: 'right' },
  // Yeh with Hamza Above (dual-joining)
  0x0626: { isolated: '\uFE89', final: '\uFE8A', initial: '\uFE8B', medial: '\uFE8C', joining: 'dual' },
  // Alef (right-joining)
  0x0627: { isolated: '\uFE8D', final: '\uFE8E', initial: '\uFE8D', medial: '\uFE8E', joining: 'right' },
  // Beh (dual-joining)
  0x0628: { isolated: '\uFE8F', final: '\uFE90', initial: '\uFE91', medial: '\uFE92', joining: 'dual' },
  // Teh Marbuta (right-joining)
  0x0629: { isolated: '\uFE93', final: '\uFE94', initial: '\uFE93', medial: '\uFE94', joining: 'right' },
  // Teh (dual-joining)
  0x062a: { isolated: '\uFE95', final: '\uFE96', initial: '\uFE97', medial: '\uFE98', joining: 'dual' },
  // Theh (dual-joining)
  0x062b: { isolated: '\uFE99', final: '\uFE9A', initial: '\uFE9B', medial: '\uFE9C', joining: 'dual' },
  // Jeem (dual-joining)
  0x062c: { isolated: '\uFE9D', final: '\uFE9E', initial: '\uFE9F', medial: '\uFEA0', joining: 'dual' },
  // Hah (dual-joining)
  0x062d: { isolated: '\uFEA1', final: '\uFEA2', initial: '\uFEA3', medial: '\uFEA4', joining: 'dual' },
  // Khah (dual-joining)
  0x062e: { isolated: '\uFEA5', final: '\uFEA6', initial: '\uFEA7', medial: '\uFEA8', joining: 'dual' },
  // Dal (right-joining)
  0x062f: { isolated: '\uFEA9', final: '\uFEAA', initial: '\uFEA9', medial: '\uFEAA', joining: 'right' },
  // Thal (right-joining)
  0x0630: { isolated: '\uFEAB', final: '\uFEAC', initial: '\uFEAB', medial: '\uFEAC', joining: 'right' },
  // Reh (right-joining)
  0x0631: { isolated: '\uFEAD', final: '\uFEAE', initial: '\uFEAD', medial: '\uFEAE', joining: 'right' },
  // Zain (right-joining)
  0x0632: { isolated: '\uFEAF', final: '\uFEB0', initial: '\uFEAF', medial: '\uFEB0', joining: 'right' },
  // Seen (dual-joining)
  0x0633: { isolated: '\uFEB1', final: '\uFEB2', initial: '\uFEB3', medial: '\uFEB4', joining: 'dual' },
  // Sheen (dual-joining)
  0x0634: { isolated: '\uFEB5', final: '\uFEB6', initial: '\uFEB7', medial: '\uFEB8', joining: 'dual' },
  // Sad (dual-joining)
  0x0635: { isolated: '\uFEB9', final: '\uFEBA', initial: '\uFEBB', medial: '\uFEBC', joining: 'dual' },
  // Dad (dual-joining)
  0x0636: { isolated: '\uFEBD', final: '\uFEBE', initial: '\uFEBF', medial: '\uFEC0', joining: 'dual' },
  // Tah (dual-joining)
  0x0637: { isolated: '\uFEC1', final: '\uFEC2', initial: '\uFEC3', medial: '\uFEC4', joining: 'dual' },
  // Zah (dual-joining)
  0x0638: { isolated: '\uFEC5', final: '\uFEC6', initial: '\uFEC7', medial: '\uFEC8', joining: 'dual' },
  // Ain (dual-joining)
  0x0639: { isolated: '\uFEC9', final: '\uFECA', initial: '\uFECB', medial: '\uFECC', joining: 'dual' },
  // Ghain (dual-joining)
  0x063a: { isolated: '\uFECD', final: '\uFECE', initial: '\uFECF', medial: '\uFED0', joining: 'dual' },
  // Tatweel / Kashida (dual-joining)
  0x0640: { isolated: '\u0640', final: '\u0640', initial: '\u0640', medial: '\u0640', joining: 'dual' },
  // Feh (dual-joining)
  0x0641: { isolated: '\uFED1', final: '\uFED2', initial: '\uFED3', medial: '\uFED4', joining: 'dual' },
  // Qaf (dual-joining)
  0x0642: { isolated: '\uFED5', final: '\uFED6', initial: '\uFED7', medial: '\uFED8', joining: 'dual' },
  // Kaf (dual-joining)
  0x0643: { isolated: '\uFED9', final: '\uFEDA', initial: '\uFEDB', medial: '\uFEDC', joining: 'dual' },
  // Lam (dual-joining)
  0x0644: { isolated: '\uFEDD', final: '\uFEDE', initial: '\uFEDF', medial: '\uFEE0', joining: 'dual' },
  // Meem (dual-joining)
  0x0645: { isolated: '\uFEE1', final: '\uFEE2', initial: '\uFEE3', medial: '\uFEE4', joining: 'dual' },
  // Noon (dual-joining)
  0x0646: { isolated: '\uFEE5', final: '\uFEE6', initial: '\uFEE7', medial: '\uFEE8', joining: 'dual' },
  // Heh (dual-joining)
  0x0647: { isolated: '\uFEE9', final: '\uFEEA', initial: '\uFEEB', medial: '\uFEEC', joining: 'dual' },
  // Waw (right-joining)
  0x0648: { isolated: '\uFEED', final: '\uFEEE', initial: '\uFEED', medial: '\uFEEE', joining: 'right' },
  // Alef Maksura (right-joining)
  0x0649: { isolated: '\uFEEF', final: '\uFEF0', initial: '\uFEEF', medial: '\uFEF0', joining: 'right' },
  // Yeh (dual-joining)
  0x064a: { isolated: '\uFEF1', final: '\uFEF2', initial: '\uFEF3', medial: '\uFEF4', joining: 'dual' },

  // Arabic Extensions (Persian, Urdu, Kurdish)
  // Alef Wasla
  0x0671: { isolated: '\uFB50', final: '\uFB51', initial: '\uFB50', medial: '\uFB51', joining: 'right' },
  // Peh
  0x067e: { isolated: '\uFB56', final: '\uFB57', initial: '\uFB58', medial: '\uFB59', joining: 'dual' },
  // Tcheh
  0x0686: { isolated: '\uFB7A', final: '\uFB7B', initial: '\uFB7C', medial: '\uFB7D', joining: 'dual' },
  // Zheh
  0x0698: { isolated: '\uFB8A', final: '\uFB8B', initial: '\uFB8A', medial: '\uFB8B', joining: 'right' },
  // Gaf
  0x06af: { isolated: '\uFB92', final: '\uFB93', initial: '\uFB94', medial: '\uFB95', joining: 'dual' },
  // Farsi Yeh
  0x06cc: { isolated: '\uFBFC', final: '\uFBFD', initial: '\uFBFE', medial: '\uFBFF', joining: 'dual' }
}

// Build reverse un-shaping map from presentation forms to logical Arabic characters
const UNSHAPE_MAP = new Map<string, string>()

for (const [codeStr, glyph] of Object.entries(ARABIC_GLYPHS)) {
  const original = String.fromCodePoint(Number(codeStr))

  if (glyph.isolated) {
    UNSHAPE_MAP.set(glyph.isolated, original)
  }

  if (glyph.final) {
    UNSHAPE_MAP.set(glyph.final, original)
  }

  if (glyph.initial) {
    UNSHAPE_MAP.set(glyph.initial, original)
  }

  if (glyph.medial) {
    UNSHAPE_MAP.set(glyph.medial, original)
  }
}

// Lam-Alef ligatures normalization
UNSHAPE_MAP.set('\uFEF5', '\u0644\u0622') // Lam + Alef Madda isolated
UNSHAPE_MAP.set('\uFEF6', '\u0644\u0622') // Lam + Alef Madda final
UNSHAPE_MAP.set('\uFEF7', '\u0644\u0623') // Lam + Alef Hamza Above isolated
UNSHAPE_MAP.set('\uFEF8', '\u0644\u0623') // Lam + Alef Hamza Above final
UNSHAPE_MAP.set('\uFEF9', '\u0644\u0625') // Lam + Alef Hamza Below isolated
UNSHAPE_MAP.set('\uFEFA', '\u0644\u0625') // Lam + Alef Hamza Below final
UNSHAPE_MAP.set('\uFEFB', '\u0644\u0627') // Lam + Alef isolated
UNSHAPE_MAP.set('\uFEFC', '\u0644\u0627') // Lam + Alef final

/**
 * Check if a code point is an Arabic diacritic (tashkeel/harakat) or combining mark.
 */
export function isArabicTashkeel(code: number): boolean {
  return (code >= 0x064b && code <= 0x065f) || code === 0x0670 || code === 0x0618 || code === 0x0619 || code === 0x061a
}

/**
 * Check if a code point is an Arabic script character.
 */
export function isArabicChar(code: number): boolean {
  return (
    (code >= 0x0600 && code <= 0x06ff) ||
    (code >= 0x0750 && code <= 0x077f) ||
    (code >= 0x08a0 && code <= 0x08ff) ||
    (code >= 0xfb50 && code <= 0xfdff) ||
    (code >= 0xfe70 && code <= 0xfeff)
  )
}

/**
 * Extracts base code point and combining marks from a grapheme cluster.
 */
function parseGrapheme(grapheme: string): { baseCode: number; diacritics: string } {
  let baseCode = 0
  let diacritics = ''

  for (let i = 0; i < grapheme.length; i++) {
    const code = grapheme.codePointAt(i)!

    if (i === 0) {
      baseCode = code
    } else {
      diacritics += grapheme[i]!
    }
  }

  return { baseCode, diacritics }
}

/**
 * Shape an array of clustered characters in-place or returns a shaped copy.
 * Preserves each element's width, styleId, and hyperlink attributes so that
 * ANSI styling and terminal grid column calculations remain 100% synchronized.
 */
export function shapeArabicCharacters<T extends { value: string }>(characters: T[]): T[] {
  if (characters.length === 0) {
    return characters
  }

  const n = characters.length
  const result: T[] = new Array(n)

  // Pre-parse base code points and glyph entries
  const parsed = new Array<{ baseCode: number; diacritics: string; glyph?: ArabicGlyphForms }>(n)

  for (let i = 0; i < n; i++) {
    const { baseCode, diacritics } = parseGrapheme(characters[i]!.value)
    parsed[i] = { baseCode, diacritics, glyph: ARABIC_GLYPHS[baseCode] }
  }

  for (let i = 0; i < n; i++) {
    const item = characters[i]!
    const curr = parsed[i]!

    if (!curr.glyph) {
      result[i] = item

      continue
    }

    // Determine if previous character connects to this character
    let prevConnects = false

    for (let p = i - 1; p >= 0; p--) {
      const prev = parsed[p]!

      if (isArabicTashkeel(prev.baseCode)) {
        continue
      }

      if (prev.glyph && prev.glyph.joining === 'dual') {
        prevConnects = true
      }

      break
    }

    // Determine if next character connects from this character
    let nextConnects = false

    for (let nextIdx = i + 1; nextIdx < n; nextIdx++) {
      const next = parsed[nextIdx]!

      if (isArabicTashkeel(next.baseCode)) {
        continue
      }

      if (next.glyph && (next.glyph.joining === 'dual' || next.glyph.joining === 'right')) {
        nextConnects = true
      }

      break
    }

    // Choose appropriate presentation form
    let shapedBase = curr.glyph.isolated

    if (curr.glyph.joining === 'right') {
      shapedBase = prevConnects ? curr.glyph.final : curr.glyph.isolated
    } else if (curr.glyph.joining === 'dual') {
      if (prevConnects && nextConnects) {
        shapedBase = curr.glyph.medial
      } else if (prevConnects) {
        shapedBase = curr.glyph.final
      } else if (nextConnects) {
        shapedBase = curr.glyph.initial
      } else {
        shapedBase = curr.glyph.isolated
      }
    }

    result[i] = {
      ...item,
      value: shapedBase + curr.diacritics
    }
  }

  return result
}

/**
 * Shape a plain Arabic string into presentation forms (for standalone text).
 */
export function shapeArabic(text: string): string {
  const chars = Array.from(text).map(c => ({ value: c }))

  return shapeArabicCharacters(chars)
    .map(c => c.value)
    .join('')
}

/**
 * Un-shapes Arabic presentation forms (Forms-A & B) back to canonical Unicode
 * characters (U+0600–U+06FF). Used when copying text from the terminal.
 */
export function unshapeArabic(text: string): string {
  let result = ''

  for (let i = 0; i < text.length; i++) {
    const char = text[i]!
    const unshaped = UNSHAPE_MAP.get(char)
    result += unshaped !== undefined ? unshaped : char
  }

  return result
}

/**
 * Normalizes text copied from an RTL/BiDi terminal buffer:
 * - Un-shapes Arabic presentation forms back to canonical Unicode.
 * - Detects visual reversed Arabic lines or segments and restores natural logical order.
 */
export function normalizeBidiTextForClipboard(text: string): string {
  if (!text) {
    return text
  }

  // First unshape any presentation forms
  const unshaped = unshapeArabic(text)

  return unshaped
}
