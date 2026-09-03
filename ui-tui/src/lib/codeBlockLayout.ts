// Shared layout helpers for the fenced-code panel renderer
// (`components/markdown.tsx::CodeBlock`) and its virtual-height estimator
// (`lib/virtualHeights.ts::estimateBodyHeight`). Keeping these in one place
// keeps the two views from drifting on width budget, mode threshold, and
// fence detection.

import { stringWidth } from '@hermes/ink'

// Below this body width the rounded outline costs more than half the row
// in border + padding cells; the renderer falls back to a left-accent view
// and the estimator must follow the same threshold.
export const CODE_PANEL_MIN_WIDTH = 20

// Normal mode: 1 left border + 1 right border + 1 left padding + 1 right
// padding. Narrow / compact: 1 left border + 1 paddingLeft.
const NORMAL_PANEL_OVERHEAD = 4
const NARROW_PANEL_OVERHEAD = 2

export const isNarrowPanel = (cols: number, compact: boolean): boolean =>
  compact || cols < CODE_PANEL_MIN_WIDTH

export const innerContentWidth = (cols: number, compact: boolean): number => {
  const overhead = isNarrowPanel(cols, compact) ? NARROW_PANEL_OVERHEAD : NORMAL_PANEL_OVERHEAD

  return Math.max(1, cols - overhead)
}

// Ink's border-embedding path (`packages/hermes-ink/src/ink/render-border.ts`)
// takes a JS-substring fallback when `stringWidth(text) >= borderLength - 2`.
// `borderText.content` here is ` ${label} `, so two cells are the
// surrounding spaces and one is the `╭` corner. The trimmed label budget
// is therefore `cols - 5`. Truncation must keep the post-truncation result
// (ellipsis included) inside this budget.
export const borderLabelWidth = (cols: number): number => Math.max(0, cols - 5)

// Renderer- and estimator-facing fence regex. The renderer's own local
// copies of these patterns were removed so both call sites share one
// definition.
export const FENCE_OPEN_RE = /^\s*(`{3,}|~{3,})(.*)$/
export const FENCE_CLOSE_RE = /^\s*(`{3,}|~{3,})\s*$/

// Grapheme-safe truncation. `ellipsis` (default `…`) is appended only when
// truncation occurs and its width is *included* in the budget, so the
// post-truncation result always satisfies `stringWidth(result) <= maxWidth`.
// Returns '' when the budget is too small to fit the ellipsis; never
// produces broken surrogate pairs or replacement characters.
export const truncateToWidth = (text: string, maxWidth: number, ellipsis: string = '…'): string => {
  if (!text) {
    return ''
  }

  if (maxWidth <= 0) {
    return ''
  }

  if (stringWidth(text) <= maxWidth) {
    return text
  }

  const ellipsisWidth = stringWidth(ellipsis)

  if (ellipsisWidth > maxWidth) {
    return ''
  }

  const budget = maxWidth - ellipsisWidth

  const segments =
    typeof Intl !== 'undefined' && 'Segmenter' in Intl
      ? new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(text)
      : null

  let out = ''

  if (segments) {
    for (const { segment: g } of segments) {
      if (stringWidth(out + g) > budget) {
        break
      }

      out += g
    }
  } else {
    for (const g of Array.from(text)) {
      if (stringWidth(out + g) > budget) {
        break
      }

      out += g
    }
  }


  return out + ellipsis
}

// Chrome rows the renderer adds on top of the wrapped code rows. Normal
// panel: top + bottom border. Narrow / compact: optional language row
// only — no top/bottom border. The renderer always emits an empty content
// row for an empty fence, so callers add at least 1 wrapped row for the
// body before adding this.
export const chromeRows = (bodyWidth: number, compact: boolean, hasLang: boolean): number =>
  isNarrowPanel(bodyWidth, compact) ? (hasLang ? 1 : 0) : 2
