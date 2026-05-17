import { stringWidth } from '@hermes/ink'
import wrapAnsi from 'wrap-ansi'

import type { Role } from '../types.js'

export const COMPOSER_PROMPT_GAP_WIDTH = 1

let _seg: Intl.Segmenter | null = null
const seg = () => (_seg ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' }))

interface VisualLine {
  end: number
  start: number
}

const graphemes = (value: string) =>
  [...seg().segment(value)].map(({ segment, index }) => ({
    end: index + segment.length,
    index,
    segment,
    width: Math.max(1, stringWidth(segment))
  }))

// Build VisualLines from wrap-ansi's output by mapping each emitted character
// back to its original offset in `value`. wrap-ansi only INSERTS '\n' at wrap
// boundaries — it never drops, reorders, or substitutes existing characters —
// so a parallel walk uniquely identifies each line's source range.
//
// This used to be a hand-rolled word-wrap whose break points disagreed with
// wrap-ansi in subtle but visible ways: exact-fill rows pushed the cursor to
// a phantom next line, mid-word breaks landed one grapheme off, etc. The
// composer's TextInput renders text via Ink's <Text wrap="wrap">, which
// delegates to wrap-ansi — so any drift between the two algorithms parks the
// hardware cursor several cells away from the last rendered character.
// Sourcing both from wrap-ansi guarantees agreement.
function visualLines(value: string, cols: number): VisualLine[] {
  if (!value.length) {
    return [{ start: 0, end: 0 }]
  }

  const width = Math.max(1, cols)
  const wrapped = wrapAnsi(value, width, { hard: true, trim: false })
  const lines: VisualLine[] = []

  let originalIdx = 0
  let lineStart = 0

  for (let i = 0; i < wrapped.length; i += 1) {
    const ch = wrapped[i]!

    if (ch === '\n') {
      // wrap-ansi inserts '\n' to mark a soft-wrap boundary OR copies a
      // literal '\n' from the input. Either way the next char in `wrapped`
      // begins a new visual line. If the source character is a hard '\n',
      // consume it (it doesn't appear in either line). Otherwise the '\n'
      // is purely a wrap marker and originalIdx stays put.
      lines.push({ start: lineStart, end: originalIdx })
      const isHardNewline = originalIdx < value.length && value[originalIdx] === '\n'

      if (isHardNewline) {
        originalIdx += 1
      }

      lineStart = originalIdx
      continue
    }

    // Defensive sync check. wrap-ansi (with `hard: true, trim: false`, no
    // styled input) is documented to only insert '\n' at break points and
    // never substitute, drop, or reorder source characters — so under those
    // options `wrapped[i]` should always equal `value[originalIdx]`. But
    // future option changes, library upgrades, or callers that start passing
    // styled input (ANSI escapes) could violate that invariant silently. If
    // they do, we'd slide `originalIdx` past the end of `value` and emit
    // garbage line ranges with no diagnostic. Realign by scanning forward
    // for the matching character; bail out (return whatever we have) if the
    // sync is unrecoverable rather than producing wrong-but-plausible output.
    if (originalIdx >= value.length) {
      break
    }

    if (value[originalIdx] !== ch) {
      const reSync = value.indexOf(ch, originalIdx)

      if (reSync === -1) {
        break
      }

      originalIdx = reSync
    }

    originalIdx += 1
  }

  lines.push({ start: lineStart, end: originalIdx })

  // wrap-ansi collapses an empty input into [""] which we already handled
  // above; preserve the invariant that lines is never empty for any input.
  return lines.length ? lines : [{ start: 0, end: 0 }]
}

function widthBetween(value: string, start: number, end: number) {
  let width = 0

  for (const part of graphemes(value.slice(start, end))) {
    width += part.width
  }

  return width
}

/**
 * Mirrors the word-wrap behavior used by the composer TextInput.
 * Returns the zero-based visual line and column of the cursor cell.
 *
 * IMPORTANT: this MUST stay in lock-step with how Ink's `<Text wrap="wrap">`
 * lays the value out (which uses `wrap-ansi`). Any divergence parks the
 * hardware cursor several cells off the last rendered character — see the
 * "cursor drift past blank cells" bug. visualLinesFromWrappedOutput is
 * sourced directly from wrap-ansi to enforce that invariant.
 */
export function cursorLayout(value: string, cursor: number, cols: number) {
  const pos = Math.max(0, Math.min(cursor, value.length))
  const w = Math.max(1, cols)
  const lines = visualLines(value, w)
  let lineIndex = 0

  for (let i = 0; i < lines.length; i += 1) {
    if (lines[i]!.start <= pos) {
      lineIndex = i
    } else {
      break
    }
  }

  const line = lines[lineIndex]!
  const column = widthBetween(value, line.start, Math.min(pos, line.end))

  // NOTE: the previous implementation forced an extra line break when
  // `column >= w` (the "trailing cursor-cell overflows" rule). With
  // visualLinesFromWrappedOutput sourcing breaks from wrap-ansi, the line
  // wrapping above already matches what Ink will actually render. Pushing
  // the cursor onto a phantom next line here would re-introduce the same
  // drift we're fixing, so we don't.
  return { column, line: lineIndex }
}

export function offsetFromPosition(value: string, row: number, col: number, cols: number) {
  if (!value.length) {
    return 0
  }

  const lines = visualLines(value, cols)
  const target = lines[Math.max(0, Math.min(lines.length - 1, Math.floor(row)))]!
  const targetCol = Math.max(0, Math.floor(col))
  let column = 0

  for (const part of graphemes(value.slice(target.start, target.end))) {
    if (targetCol <= column + Math.max(0, part.width - 1)) {
      return target.start + part.index
    }

    column += part.width
  }

  return target.end
}

export function inputVisualHeight(value: string, columns: number) {
  return cursorLayout(value, value.length, columns).line + 1
}

export function composerPromptWidth(promptText: string) {
  return Math.max(1, stringWidth(promptText)) + COMPOSER_PROMPT_GAP_WIDTH
}

export function transcriptGutterWidth(role: Role, userPrompt: string) {
  return role === 'user' ? composerPromptWidth(userPrompt) : 3
}

export function transcriptBodyWidth(totalCols: number, role: Role, userPrompt: string) {
  return Math.max(20, totalCols - transcriptGutterWidth(role, userPrompt) - 2)
}

export function stableComposerColumns(totalCols: number, promptWidth: number) {
  // Physical render/wrap width. Always reserve outer composer padding and
  // prompt prefix. Only reserve the transcript scrollbar gutter when the
  // terminal is wide enough; on narrow panes, preserving input columns beats
  // keeping gutters visually aligned.
  return Math.max(1, totalCols - promptWidth - 2 - (totalCols - promptWidth >= 24 ? 2 : 0))
}
