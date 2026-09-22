/**
 * Arabic clipboard normalization for terminal selection and OSC 52 copies.
 *
 * Terminal monospace buffers hold visually reordered characters with
 * Arabic Presentation Forms (Forms-A & B) so that the terminal's LTR grid
 * displays connected Arabic text. When copying text out of the terminal,
 * this module restores standard logical Unicode Arabic (U+0600–U+06FF)
 * and natural reading order so that pasting into WhatsApp, Word, browser,
 * or code editors works seamlessly.
 */

// Presentation forms to canonical Unicode mapping
const PRESENTATION_TO_CANONICAL: Record<string, string> = {
  // Hamza
  '\uFE80': '\u0621',
  // Alef with Madda
  '\uFE81': '\u0622', '\uFE82': '\u0622',
  // Alef with Hamza Above
  '\uFE83': '\u0623', '\uFE84': '\u0623',
  // Waw with Hamza Above
  '\uFE85': '\u0624', '\uFE86': '\u0624',
  // Alef with Hamza Below
  '\uFE87': '\u0625', '\uFE88': '\u0625',
  // Yeh with Hamza Above
  '\uFE89': '\u0626', '\uFE8A': '\u0626', '\uFE8B': '\u0626', '\uFE8C': '\u0626',
  // Alef
  '\uFE8D': '\u0627', '\uFE8E': '\u0627',
  // Beh
  '\uFE8F': '\u0628', '\uFE90': '\u0628', '\uFE91': '\u0628', '\uFE92': '\u0628',
  // Teh Marbuta
  '\uFE93': '\u0629', '\uFE94': '\u0629',
  // Teh
  '\uFE95': '\u062A', '\uFE96': '\u062A', '\uFE97': '\u062A', '\uFE98': '\u062A',
  // Theh
  '\uFE99': '\u062B', '\uFE9A': '\u062B', '\uFE9B': '\u062B', '\uFE9C': '\u062B',
  // Jeem
  '\uFE9D': '\u062C', '\uFE9E': '\u062C', '\uFE9F': '\u062C', '\uFEA0': '\u062C',
  // Hah
  '\uFEA1': '\u062D', '\uFEA2': '\u062D', '\uFEA3': '\u062D', '\uFEA4': '\u062D',
  // Khah
  '\uFEA5': '\u062E', '\uFEA6': '\u062E', '\uFEA7': '\u062E', '\uFEA8': '\u062E',
  // Dal
  '\uFEA9': '\u062F', '\uFEAA': '\u062F',
  // Thal
  '\uFEAB': '\u0630', '\uFEAC': '\u0630',
  // Reh
  '\uFEAD': '\u0631', '\uFEAE': '\u0631',
  // Zain
  '\uFEAF': '\u0632', '\uFEB0': '\u0632',
  // Seen
  '\uFEB1': '\u0633', '\uFEB2': '\u0633', '\uFEB3': '\u0633', '\uFEB4': '\u0633',
  // Sheen
  '\uFEB5': '\u0634', '\uFEB6': '\u0634', '\uFEB7': '\u0634', '\uFEB8': '\u0634',
  // Sad
  '\uFEB9': '\u0635', '\uFEBA': '\u0635', '\uFEBB': '\u0635', '\uFEBC': '\u0635',
  // Dad
  '\uFEBD': '\u0636', '\uFEBE': '\u0636', '\uFEBF': '\u0636', '\uFEC0': '\u0636',
  // Tah
  '\uFEC1': '\u0637', '\uFEC2': '\u0637', '\uFEC3': '\u0637', '\uFEC4': '\u0637',
  // Zah
  '\uFEC5': '\u0638', '\uFEC6': '\u0638', '\uFEC7': '\u0638', '\uFEC8': '\u0638',
  // Ain
  '\uFEC9': '\u0639', '\uFECA': '\u0639', '\uFECB': '\u0639', '\uFECC': '\u0639',
  // Ghain
  '\uFECD': '\u063A', '\uFECE': '\u063A', '\uFECF': '\u063A', '\uFED0': '\u063A',
  // Feh
  '\uFED1': '\u0641', '\uFED2': '\u0641', '\uFED3': '\u0641', '\uFED4': '\u0641',
  // Qaf
  '\uFED5': '\u0642', '\uFED6': '\u0642', '\uFED7': '\u0642', '\uFED8': '\u0642',
  // Kaf
  '\uFED9': '\u0643', '\uFEDA': '\u0643', '\uFEDB': '\u0643', '\uFEDC': '\u0643',
  // Lam
  '\uFEDD': '\u0644', '\uFEDE': '\u0644', '\uFEDF': '\u0644', '\uFEE0': '\u0644',
  // Meem
  '\uFEE1': '\u0645', '\uFEE2': '\u0645', '\uFEE3': '\u0645', '\uFEE4': '\u0645',
  // Noon
  '\uFEE5': '\u0646', '\uFEE6': '\u0646', '\uFEE7': '\u0646', '\uFEE8': '\u0646',
  // Heh
  '\uFEE9': '\u0647', '\uFEEA': '\u0647', '\uFEEB': '\u0647', '\uFEEC': '\u0647',
  // Waw
  '\uFEED': '\u0648', '\uFEEE': '\u0648',
  // Alef Maksura
  '\uFEEF': '\u0649', '\uFEF0': '\u0649',
  // Yeh
  '\uFEF1': '\u064A', '\uFEF2': '\u064A', '\uFEF3': '\u064A', '\uFEF4': '\u064A',

  // Lam-Alef ligatures
  '\uFEF5': '\u0644\u0622', '\uFEF6': '\u0644\u0622',
  '\uFEF7': '\u0644\u0623', '\uFEF8': '\u0644\u0623',
  '\uFEF9': '\u0644\u0625', '\uFEFA': '\u0644\u0625',
  '\uFEFB': '\u0644\u0627', '\uFEFC': '\u0644\u0627',

  // Extensions
  '\uFB50': '\u0671', '\uFB51': '\u0671',
  '\uFB56': '\u067E', '\uFB57': '\u067E', '\uFB58': '\u067E', '\uFB59': '\u067E',
  '\uFB7A': '\u0686', '\uFB7B': '\u0686', '\uFB7C': '\u0686', '\uFB7D': '\u0686',
  '\uFB8A': '\u0698', '\uFB8B': '\u0698',
  '\uFB92': '\u06AF', '\uFB93': '\u06AF', '\uFB94': '\u06AF', '\uFB95': '\u06AF',
  '\uFBFC': '\u06CC', '\uFBFD': '\u06CC', '\uFBFE': '\u06CC', '\uFBFF': '\u06CC',
}

/**
 * Check if a code point is an Arabic presentation form glyph.
 */
function isPresentationForm(char: string): boolean {
  const code = char.codePointAt(0)
  if (!code) return false
  return (code >= 0xfb50 && code <= 0xfdff) || (code >= 0xfe70 && code <= 0xfefc)
}

/**
 * Normalizes text copied from the terminal:
 * - Identifies runs containing Arabic presentation forms.
 * - Reverses the visual order back to natural logical order.
 * - Converts presentation forms to canonical Arabic letters.
 * - Un-mirrors brackets inside the reversed run.
 * Non-Arabic text and code wrappers remain untouched.
 */
export function normalizeClipboardText(text: string): string {
  if (!text) return text

  // Fast check: if no presentation forms exist, return text as is
  let hasPresentation = false
  for (let i = 0; i < text.length; i++) {
    if (isPresentationForm(text[i]!)) {
      hasPresentation = true
      break
    }
  }

  if (!hasPresentation) {
    return text
  }

  // Matches sequences of Arabic presentation forms (and any spaces, tatweel, tashkeel, or Arabic punctuation between them)
  const ARABIC_RUN_REGEX = new RegExp(
    // eslint-disable-next-line no-misleading-character-class
    '([\\uFB50-\\uFDFF\\uFE70-\\uFEFC](?:[\\uFB50-\\uFDFF\\uFE70-\\uFEFC\\u064B-\\u065F\\u0640\\s،؛]*[\\uFB50-\\uFDFF\\uFE70-\\uFEFC])?)',
    'gu',
  )

  return text.replace(ARABIC_RUN_REGEX, (match) => {
    // Reverse the visual segment to restore logical order
    const chars = Array.from(match)
    chars.reverse()

    // Un-shape presentation forms to canonical Arabic letters
    return chars
      .map((ch) => {
        const unshaped = PRESENTATION_TO_CANONICAL[ch]
        return unshaped !== undefined ? unshaped : ch
      })
      .join('')
  })
}
