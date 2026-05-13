import { stringWidth } from '@hermes/ink'

import type { Role } from '../types.js'

export const COMPOSER_PROMPT_GAP_WIDTH = 1
export const COMPOSER_INPUT_MAX_VISUAL_LINES = 3
export const COMPOSER_COLLAPSE_MARKER = '[[...]]'

let _seg: Intl.Segmenter | null = null
const seg = () => (_seg ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' }))

interface VisualLine {
  end: number
  start: number
}

interface ProjectedSegment {
  displayEnd: number
  displayStart: number
  sourceEnd: number
  sourceStart: number
}

export interface ProjectedComposerInput {
  collapsed: boolean
  cursor: number
  height: number
  sourceOffsetFromDisplayOffset: (offset: number) => number
  value: string
}

const isWhitespace = (value: string) => /\s/.test(value)
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

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

function identityProjection(value: string, cursor: number, cols: number): ProjectedComposerInput {
  const pos = clamp(cursor, 0, value.length)

  return {
    collapsed: false,
    cursor: pos,
    height: inputVisualHeight(value, cols),
    sourceOffsetFromDisplayOffset: offset => clamp(offset, 0, value.length),
    value
  }
}

function projectionSourceMapper(segments: ProjectedSegment[], sourceLength: number, displayLength: number) {
  return (offset: number) => {
    const pos = clamp(offset, 0, displayLength)
    let previous: ProjectedSegment | null = null

    for (const segment of segments) {
      if (pos < segment.displayStart) {
        return clamp(segment.sourceStart, 0, sourceLength)
      }

      if (pos <= segment.displayEnd) {
        const sourceSpan = segment.sourceEnd - segment.sourceStart
        const displayDelta = clamp(pos - segment.displayStart, 0, sourceSpan)

        return clamp(segment.sourceStart + displayDelta, 0, sourceLength)
      }

      previous = segment
    }

    return clamp(previous?.sourceEnd ?? 0, 0, sourceLength)
  }
}

function linesWithTrailingCursorRow(value: string, cols: number): VisualLine[] {
  const lines = [...visualLines(value, cols)]
  const height = inputVisualHeight(value, cols)

  while (lines.length < height) {
    lines.push({ start: value.length, end: value.length })
  }

  return lines
}

export function projectComposerInput(
  value: string,
  cursor: number,
  cols: number,
  maxVisualLines = Number.POSITIVE_INFINITY
): ProjectedComposerInput {
  const width = Math.max(1, cols)
  const limit = Math.floor(maxVisualLines)
  const pos = clamp(cursor, 0, value.length)

  if (!Number.isFinite(limit) || limit < 1 || inputVisualHeight(value, width) <= limit) {
    return identityProjection(value, pos, width)
  }

  const lines = linesWithTrailingCursorRow(value, width)
  const totalLines = lines.length
  const cursorLine = clamp(cursorLayout(value, pos, width).line, 0, totalLines - 1)
  const marker = stringWidth(COMPOSER_COLLAPSE_MARKER) <= width ? COMPOSER_COLLAPSE_MARKER : '…'

  let contentStart = 0
  let contentEnd = totalLines
  let markerBefore = false
  let markerAfter = false

  if (limit === 1) {
    contentStart = cursorLine
    contentEnd = cursorLine + 1
  } else if (cursorLine < limit - 1) {
    contentStart = 0
    contentEnd = Math.min(totalLines, limit - 1)
    markerAfter = contentEnd < totalLines
  } else if (cursorLine >= totalLines - (limit - 1)) {
    contentEnd = totalLines
    contentStart = Math.max(0, totalLines - (limit - 1))
    markerBefore = contentStart > 0
  } else {
    const contentSlots = Math.max(1, limit - 2)
    const beforeCursor = Math.floor((contentSlots - 1) / 2)

    contentStart = clamp(cursorLine - beforeCursor, 1, totalLines - contentSlots - 1)
    contentEnd = contentStart + contentSlots
    markerBefore = true
    markerAfter = true
  }

  const renderedLines: { isMarker?: boolean; sourceEnd: number; sourceStart: number; text: string }[] = []

  if (markerBefore) {
    const sourceStart = lines[contentStart]?.start ?? 0
    renderedLines.push({ isMarker: true, sourceStart, sourceEnd: sourceStart, text: marker })
  }

  for (let i = contentStart; i < contentEnd; i += 1) {
    const line = lines[i]!
    renderedLines.push({ sourceStart: line.start, sourceEnd: line.end, text: value.slice(line.start, line.end) })
  }

  if (markerAfter) {
    const sourceEnd = lines[contentEnd - 1]?.end ?? value.length
    renderedLines.push({ isMarker: true, sourceStart: sourceEnd, sourceEnd, text: marker })
  }

  let projectedValue = ''
  let projectedCursor: number | null = null
  const segments: ProjectedSegment[] = []
  let sourceLineIndex = contentStart

  for (let index = 0; index < renderedLines.length; index += 1) {
    const line = renderedLines[index]!
    if (index > 0) {
      projectedValue += '\n'
    }

    const displayStart = projectedValue.length
    projectedValue += line.text
    const displayEnd = projectedValue.length

    segments.push({ displayStart, displayEnd, sourceStart: line.sourceStart, sourceEnd: line.sourceEnd })

    if (!line.isMarker) {
      if (sourceLineIndex === cursorLine && projectedCursor === null) {
        projectedCursor = displayStart + clamp(pos - line.sourceStart, 0, line.text.length)
      }

      sourceLineIndex += 1
    }
  }

  const mapSource = projectionSourceMapper(segments, value.length, projectedValue.length)

  if (projectedCursor === null) {
    projectedCursor = projectedValue.length
  }

  const projectedHeight = inputVisualHeight(projectedValue, width)

  return {
    collapsed: true,
    cursor: clamp(projectedCursor, 0, projectedValue.length),
    height: Math.min(limit, projectedHeight),
    sourceOffsetFromDisplayOffset: mapSource,
    value: projectedValue
  }
}

export function projectedInputVisualHeight(value: string, columns: number, maxVisualLines: number) {
  return projectComposerInput(value, value.length, columns, maxVisualLines).height
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
