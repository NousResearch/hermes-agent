import { stringWidth } from '@hermes/ink'

import type { Role } from '../types.js'

export const COMPOSER_PROMPT_GAP_WIDTH = 1

let _seg: Intl.Segmenter | null = null
const seg = () => (_seg ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' }))

interface VisualLine {
  end: number
  start: number
}

const isWhitespace = (value: string) => /\s/.test(value)

const graphemes = (value: string) =>
  [...seg().segment(value)].map(({ segment, index }) => ({
    end: index + segment.length,
    index,
    segment,
    width: Math.max(1, stringWidth(segment))
  }))

function visualLines(value: string, cols: number): VisualLine[] {
  const width = Math.max(1, cols)
  const lines: VisualLine[] = []
  let sourceLineStart = 0

  for (const sourceLine of value.split('\n')) {
    const parts = graphemes(sourceLine)

    if (!parts.length) {
      lines.push({ start: sourceLineStart, end: sourceLineStart })
      sourceLineStart += 1
      continue
    }

    let lineStartPart = 0
    let lineStartOffset = sourceLineStart
    let column = 0
    let breakPart: null | number = null
    let i = 0

    while (i < parts.length) {
      const part = parts[i]!
      const partStart = sourceLineStart + part.index

      if (column + part.width > width && i > lineStartPart) {
        if (breakPart !== null && breakPart > lineStartPart) {
          const breakOffset = sourceLineStart + parts[breakPart - 1]!.end
          lines.push({ start: lineStartOffset, end: breakOffset })
          lineStartPart = breakPart
          lineStartOffset = breakOffset
        } else {
          lines.push({ start: lineStartOffset, end: partStart })
          lineStartPart = i
          lineStartOffset = partStart
        }

        column = 0
        breakPart = null
        i = lineStartPart
        continue
      }

      column += part.width

      if (isWhitespace(part.segment)) {
        breakPart = i + 1
      }

      i += 1

      if (column >= width && i < parts.length) {
        const next = parts[i]!
        const nextStartsWord = !isWhitespace(next.segment)

        if (breakPart !== null && breakPart > lineStartPart && nextStartsWord) {
          const breakOffset = sourceLineStart + parts[breakPart - 1]!.end
          lines.push({ start: lineStartOffset, end: breakOffset })
          lineStartPart = breakPart
          lineStartOffset = breakOffset
          column = 0
          breakPart = null
          i = lineStartPart
        }
      }
    }

    lines.push({ start: lineStartOffset, end: sourceLineStart + sourceLine.length })
    sourceLineStart += sourceLine.length + 1
  }

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
  let column = widthBetween(value, line.start, Math.min(pos, line.end))

  // trailing cursor-cell overflows to the next row at the wrap column
  if (column >= w) {
    lineIndex++
    column = 0
  }

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
