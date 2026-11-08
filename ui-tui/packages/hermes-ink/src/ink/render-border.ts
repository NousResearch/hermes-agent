import chalk from 'chalk'
import cliBoxes, { type Boxes, type BoxStyle } from 'cli-boxes'

import sliceAnsi from '../utils/sliceAnsi.js'

import { applyColor } from './colorize.js'
import type { DOMNode } from './dom.js'
import type Output from './output.js'
import { stringWidth } from './stringWidth.js'
import type { Color } from './styles.js'

export type BorderTextOptions = {
  content: string // Pre-rendered string with ANSI color codes
  position: 'top' | 'bottom'
  align: 'start' | 'end' | 'center'
  offset?: number // Only used with 'start' or 'end' alignment. Number of characters from the edge.
}

export const CUSTOM_BORDER_STYLES = {
  dashed: {
    top: '╌',
    left: '╎',
    right: '╎',
    bottom: '╌',
    // there aren't any line-drawing characters for dashes unfortunately
    topLeft: ' ',
    topRight: ' ',
    bottomLeft: ' ',
    bottomRight: ' '
  }
} as const

export type BorderStyle = keyof Boxes | keyof typeof CUSTOM_BORDER_STYLES | BoxStyle

function styleBorderLine(line: string, color: Color | undefined, dim: boolean | undefined): string {
  let styled = applyColor(line, color)

  if (dim) {
    styled = chalk.dim(styled)
  }

  return styled
}

function sliceAnsiToWidth(text: string, start: number, end: number): { leadingColumns: number; text: string } {
  // sliceAnsi omits a wide grapheme bisected by `start`; retain its partial
  // cell as blank space so later graphemes keep their source coordinates.
  const leadingColumns = Math.max(0, stringWidth(sliceAnsi(text, 0, start)) - start)
  let sliced = sliceAnsi(text, start, end)

  if (stringWidth(sliced) > end - start - leadingColumns) {
    sliced = sliceAnsi(text, start, end - 1)
  }

  return { leadingColumns, text: sliced }
}

function borderRun(
  start: number,
  end: number,
  borderLength: number,
  borderChar: string,
  startCorner: string,
  endCorner: string
): string {
  if (start >= end) {
    return ''
  }

  const includeStartCorner = start === 0 && startCorner.length > 0
  const includeEndCorner = end === borderLength && endCorner.length > 0
  const repeated = Math.max(0, end - start - (includeStartCorner ? 1 : 0) - (includeEndCorner ? 1 : 0))

  return (includeStartCorner ? startCorner : '') + borderChar.repeat(repeated) + (includeEndCorner ? endCorner : '')
}

function renderHorizontalBorder(
  x: number,
  y: number,
  width: number,
  output: Output,
  visibleX1: number,
  visibleX2: number,
  borderChar: string,
  startCorner: string,
  endCorner: string,
  color: Color | undefined,
  dim: boolean | undefined,
  borderText: BorderTextOptions | undefined
): void {
  const borderLength =
    Math.max(0, width - (startCorner ? 1 : 0) - (endCorner ? 1 : 0)) + (startCorner ? 1 : 0) + (endCorner ? 1 : 0)

  const clippedX1 = Math.max(0, Math.floor(visibleX1))
  const clippedX2 = Math.min(output.width, Math.ceil(visibleX2))
  const sliceStart = Math.max(0, Math.ceil(clippedX1 - x))
  const sliceEnd = Math.min(borderLength, Math.ceil(clippedX2 - x))

  if (
    !Number.isSafeInteger(width) ||
    width < 0 ||
    !Number.isSafeInteger(sliceStart) ||
    !Number.isSafeInteger(sliceEnd) ||
    sliceStart >= sliceEnd
  ) {
    return
  }

  const writeBorderRun = (start: number, end: number) => {
    const text = borderRun(start, end, borderLength, borderChar, startCorner, endCorner)

    if (text) {
      output.write(x + start, y, styleBorderLine(text, color, dim))
    }
  }

  if (!borderText) {
    writeBorderRun(sliceStart, sliceEnd)

    return
  }

  const textLength = stringWidth(borderText.content)

  if (textLength >= borderLength - 2) {
    const { leadingColumns, text } = sliceAnsiToWidth(borderText.content, sliceStart, sliceEnd)

    if (text) {
      output.write(x + sliceStart + leadingColumns, y, text)
    }

    return
  }

  let position: number

  if (borderText.align === 'center') {
    position = Math.floor((borderLength - textLength) / 2)
  } else if (borderText.align === 'start') {
    position = (borderText.offset ?? 0) + 1
  } else {
    position = borderLength - textLength - (borderText.offset ?? 0) - 1
  }

  position = Math.max(1, Math.min(position, borderLength - textLength - 1))

  writeBorderRun(sliceStart, Math.min(sliceEnd, position))

  const visibleTextStart = Math.max(sliceStart, position)
  const visibleTextEnd = Math.min(sliceEnd, position + textLength)

  if (visibleTextStart < visibleTextEnd) {
    const { leadingColumns, text } = sliceAnsiToWidth(
      borderText.content,
      visibleTextStart - position,
      visibleTextEnd - position
    )

    if (text) {
      output.write(x + visibleTextStart + leadingColumns, y, text)
    }
  }

  writeBorderRun(Math.max(sliceStart, position + textLength), sliceEnd)
}

const renderBorder = (
  x: number,
  y: number,
  node: DOMNode,
  output: Output,
  visibleX1 = 0,
  visibleX2 = output.width,
  visibleY1 = 0,
  visibleY2 = output.height
): void => {
  if (node.style.borderStyle) {
    const width = Math.floor(node.yogaNode!.getComputedWidth())
    const height = Math.floor(node.yogaNode!.getComputedHeight())

    const box =
      typeof node.style.borderStyle === 'string'
        ? (CUSTOM_BORDER_STYLES[node.style.borderStyle as keyof typeof CUSTOM_BORDER_STYLES] ??
          cliBoxes[node.style.borderStyle as keyof Boxes])
        : node.style.borderStyle

    const topBorderColor = node.style.borderTopColor ?? node.style.borderColor

    const bottomBorderColor = node.style.borderBottomColor ?? node.style.borderColor

    const leftBorderColor = node.style.borderLeftColor ?? node.style.borderColor

    const rightBorderColor = node.style.borderRightColor ?? node.style.borderColor

    const dimTopBorderColor = node.style.borderTopDimColor ?? node.style.borderDimColor

    const dimBottomBorderColor = node.style.borderBottomDimColor ?? node.style.borderDimColor

    const dimLeftBorderColor = node.style.borderLeftDimColor ?? node.style.borderDimColor

    const dimRightBorderColor = node.style.borderRightDimColor ?? node.style.borderDimColor

    const showTopBorder = node.style.borderTop !== false
    const showBottomBorder = node.style.borderBottom !== false
    const showLeftBorder = node.style.borderLeft !== false
    const showRightBorder = node.style.borderRight !== false

    const verticalTop = Math.floor(y + (showTopBorder ? 1 : 0))
    const verticalBottom = Math.ceil(y + height - (showBottomBorder ? 1 : 0))
    const clippedVerticalTop = Math.max(0, Math.floor(visibleY1), verticalTop)
    const clippedVerticalBottom = Math.min(output.height, Math.ceil(visibleY2), verticalBottom)
    const clippedVerticalHeight = Math.max(0, clippedVerticalBottom - clippedVerticalTop)

    let leftBorder = (applyColor(box.left, leftBorderColor) + '\n').repeat(clippedVerticalHeight)

    if (dimLeftBorderColor) {
      leftBorder = chalk.dim(leftBorder)
    }

    let rightBorder = (applyColor(box.right, rightBorderColor) + '\n').repeat(clippedVerticalHeight)

    if (dimRightBorderColor) {
      rightBorder = chalk.dim(rightBorder)
    }

    if (showTopBorder && y >= visibleY1 && y < visibleY2 && y >= 0 && y < output.height) {
      renderHorizontalBorder(
        x,
        y,
        width,
        output,
        visibleX1,
        visibleX2,
        box.top,
        showLeftBorder ? box.topLeft : '',
        showRightBorder ? box.topRight : '',
        topBorderColor,
        dimTopBorderColor,
        node.style.borderText?.position === 'top' ? node.style.borderText : undefined
      )
    }

    if (showLeftBorder && clippedVerticalHeight > 0 && x >= visibleX1 && x < visibleX2 && x >= 0 && x < output.width) {
      output.write(x, clippedVerticalTop, leftBorder)
    }

    const rightX = x + width - 1

    if (
      showRightBorder &&
      clippedVerticalHeight > 0 &&
      rightX >= visibleX1 &&
      rightX < visibleX2 &&
      rightX >= 0 &&
      rightX < output.width
    ) {
      output.write(rightX, clippedVerticalTop, rightBorder)
    }

    const bottomY = y + height - 1

    if (showBottomBorder && bottomY >= visibleY1 && bottomY < visibleY2 && bottomY >= 0 && bottomY < output.height) {
      renderHorizontalBorder(
        x,
        bottomY,
        width,
        output,
        visibleX1,
        visibleX2,
        box.bottom,
        showLeftBorder ? box.bottomLeft : '',
        showRightBorder ? box.bottomRight : '',
        bottomBorderColor,
        dimBottomBorderColor,
        node.style.borderText?.position === 'bottom' ? node.style.borderText : undefined
      )
    }
  }
}

export default renderBorder
