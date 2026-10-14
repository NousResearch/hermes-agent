/**
 * Pure code-fence parser for Markdown fenced code blocks.
 *
 * Recognises backtick and tilde fences (length ≥ 3), requires matching closer
 * character and minimum length, extracts info string / language, and returns
 * the raw content between the fences — exactly what should be sent to the
 * clipboard.
 */

const FENCE_OPENER_RE = /^\s*(`{3,}|~{3,})(.*)$/
const FENCE_CLOSER_RE = /^\s*(`{3,}|~{3,})\s*$/

export interface CopyBloxFence {
  /** Whether a matching closer was found. */
  closed: boolean
  /** Line index of the opening fence (0-based). */
  openLineIndex: number
  /** Index of the closing fence line, or `-1` if unclosed. */
  endLineIndex: number
  /** Opening fence character: `` ` `` or `~`. */
  fenceChar: '`' | '~'
  /** Length of the opening fence string. */
  fenceLength: number
  /** Raw info string from the opening fence line (after the fence ticks). */
  infoString: string
  /** Normalised display language (first token, lowercased). */
  language: string
  /** Exact text between the fences — what the clipboard receives. */
  rawContent: string
}

/**
 * Compute raw line boundaries in a source string.
 *
 * Returns an array of `[start, end)` offsets for each line (split on `\n`).
 * `end` is the index of the `\n` character, or `source.length` for the last
 * line if it doesn't end with a newline.
 */
function lineBoundaries(source: string): Array<[number, number]> {
  const boundaries: Array<[number, number]> = []
  let start = 0

  for (let ch = 0; ch < source.length; ch++) {
    if (source[ch] === '\n') {
      boundaries.push([start, ch])
      start = ch + 1
    }
  }

  // Last line (no trailing newline)
  if (start <= source.length) {
    boundaries.push([start, source.length])
  }

  return boundaries
}

/**
 * Parse all fenced code blocks from raw source text.
 *
 * `source` is the unmodified text from the message (before display
 * normalisation). The parser matches fences against `source` directly and
 * returns `rawContent` that is byte-accurate for clipboard use.
 */
export function parseCodeFences(source: string): CopyBloxFence[] {
  const boundaries = lineBoundaries(source)
  const fences: CopyBloxFence[] = []
  let i = 0

  while (i < boundaries.length) {
    const [lineStart, lineEnd] = boundaries[i]!
    const line = source.slice(lineStart, lineEnd)

    const match = line.match(FENCE_OPENER_RE)

    if (!match) {
      i++

      continue
    }

    const fenceChar = match[1]![0] as '`' | '~'
    const fenceLength = match[1]!.length
    const infoString = match[2]!.trim()
    const openLineIndex = i
    i++

    const contentParts: string[] = []
    let closerLine = -1

    for (; i < boundaries.length; i++) {
      const [cs, ce] = boundaries[i]!
      const cline = source.slice(cs, ce)
      const closeMatch = cline.match(FENCE_CLOSER_RE)

      if (closeMatch && closeMatch[1]![0] === fenceChar && closeMatch[1]!.length >= fenceLength) {
        closerLine = i

        break
      }

      contentParts.push(source.slice(cs, ce))
    }

    const closed = closerLine >= 0

    if (closed) {
      i++ // skip past the closer
    }

    const rawContent = contentParts.join('\n')
    const language = parseLanguage(infoString, fenceChar, rawContent)

    fences.push({
      closed,
      openLineIndex,
      endLineIndex: closerLine,
      fenceChar,
      fenceLength,
      infoString,
      language,
      rawContent
    })
  }

  return fences
}

function parseLanguage(infoString: string, fenceChar: '`' | '~', rawContent: string): string {
  if (infoString) {
    const firstToken = infoString.split(/[\s]+/)[0]!

    // Strip leading language directives like `lang: ` or `language=`
    const normalised = firstToken.toLowerCase().replace(/^language[:=]/, '')

    return normalised
  }

  // Default: if content looks like a diff, advertise that.
  if (rawContent.startsWith('--- ') || rawContent.startsWith('+++ ')) {
    return 'diff'
  }

  return 'text'
}
