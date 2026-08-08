import { TERMUX_TUI_MODE } from '../config/env.js'
import type { Msg } from '../types.js'

import {
  chromeRows,
  FENCE_CLOSE_RE,
  FENCE_OPEN_RE,
  innerContentWidth
} from './codeBlockLayout.js'
import { transcriptBodyWidth } from './inputMetrics.js'

const hashText = (text: string) => {
  let h = 5381

  for (let i = 0; i < text.length; i++) {
    h = ((h << 5) + h) ^ text.charCodeAt(i)
  }

  return (h >>> 0).toString(36)
}

export const messageHeightKey = (msg: Msg) => {
  const todoSig = msg.todos?.map(t => `${t.status}:${t.content}`).join('\u0001') ?? ''

  const panelSig =
    msg.panelData?.sections
      .map(s => `${s.title ?? ''}:${s.text?.length ?? 0}:${s.items?.length ?? 0}:${s.rows?.length ?? 0}`)
      .join('\u0001') ?? ''

  const introSig = msg.kind === 'intro' ? (msg.info?.version ?? '') : ''

  return [
    msg.role,
    msg.kind ?? '',
    hashText([msg.text, msg.thinking ?? '', msg.tools?.join('\n') ?? '', todoSig, panelSig, introSig].join('\0'))
  ].join(':')
}

// Hard cap on rows the estimator will count. Each row above this is
// invisible to the estimator (gets clipped to MAX_ESTIMATE_LINES), but
// post-mount Yoga measurement converges to the real height on first
// render. Without this, a long assistant turn (10k+ chars) costs O(text)
// per offset rebuild × every uncached item — cold-mounting a 1000-row
// transcript becomes a multi-million-char wrap walk that blocks the UI.
//
// 800 covers any realistic assistant message (the prior history-clip
// ceiling was 16 lines, then full text — this is the sane middle).
const MAX_ESTIMATE_LINES = 800

export const wrappedLines = (text: string, width: number, maxLines: number = MAX_ESTIMATE_LINES) => {
  const w = Math.max(1, width)
  // Worst case: every cell is its own row at width=1, plus a small
  // slack for the trailing partial line. Walking past this byte budget
  // cannot increase n any further once n is already past maxLines, so
  // bail. Saves O(text) walks on multi-megabyte single-line messages.
  const budget = Math.min(text.length, maxLines * w + maxLines)
  let n = 0
  let start = 0

  for (let i = 0; i <= budget; i++) {
    if (i === text.length || i === budget || text.charCodeAt(i) === 10) {
      const rows = Math.max(1, Math.ceil((i - start) / w))
      n += rows >= maxLines - n ? maxLines - n : rows
      start = i + 1

      if (n >= maxLines) {
        return maxLines
      }
    }
  }

  return n
}

// Fence-aware body height. Walks `text` linearly (no `split('\n')`, no
// pre-built fence-span list) and counts rows the way the renderer would
// paint them. For a `md` / `markdown` fence the renderer recurses with
// `<Md cols={cols}>` instead of the rounded CodeBlock panel, so its body
// is prose at the full body width and the panel chrome rows don't apply —
// the estimator must do the same. For every other language (or no
// language) the fence is the rounded panel: wrap at the panel's inner
// width and add the panel's chrome rows. Prose lines keep the original
// char-count / width formula so existing non-fence estimates are unchanged.
// The walk is bounded by the same `MAX_ESTIMATE_LINES` cap the prose
// estimator uses, so a 1M-char single line still returns in O(width) time.
const MARKDOWN_FENCE_LANGS = new Set(['md', 'markdown'])

export const estimateBodyHeight = (
  text: string,
  bodyWidth: number,
  compact: boolean,
  maxLines: number = MAX_ESTIMATE_LINES
): number => {
  if (!text) {
    return 1
  }

  const w = Math.max(1, bodyWidth)
  const innerW = innerContentWidth(bodyWidth, compact)
  const charBudget = Math.min(text.length, maxLines * w + maxLines)
  let n = 0
  let i = 0
  let lineStart = 0
  let inFence = false
  let fenceChar = ''
  let fenceLen = 0
  let fenceLang = ''
  let fenceContentStart = 0

  const addRows = (rows: number): void => {
    if (n >= maxLines) {
      return
    }

    n += Math.min(rows, maxLines - n)
  }

  // A fence body's slice ends with the newline that separates the last
  // content line from the close fence (or end-of-text for an unclosed
  // fence). `wrappedLines` would otherwise count a phantom empty line at
  // that trailing newline + the end-of-text it processes on the next
  // iteration. Strip exactly one trailing newline.
  const sliceFenceContent = (start: number, end: number): string => {
    let e = end

    if (e > start && text.charCodeAt(e - 1) === 10) {
      e--
    }

    return text.slice(start, e)
  }

  while (i <= charBudget) {
    const atEnd = i >= charBudget
    const isNL = !atEnd && i < text.length && text.charCodeAt(i) === 10

    if (atEnd || isNL) {
      const lineEnd = i
      const line = text.slice(lineStart, lineEnd)

      if (inFence) {
        const closeMatch = line.match(FENCE_CLOSE_RE)

        const isClose =
          closeMatch && closeMatch[1]![0] === fenceChar && closeMatch[1]!.length >= fenceLen

        if (isClose) {
          if (MARKDOWN_FENCE_LANGS.has(fenceLang)) {
            // `md` / `markdown` fences are recursed with `<Md>`, so the body
            // is prose at the full body width and the rounded panel's
            // chrome rows don't apply.
            addRows(wrappedLines(sliceFenceContent(fenceContentStart, lineStart), w, maxLines - n))
          } else {
            addRows(
              wrappedLines(sliceFenceContent(fenceContentStart, lineStart), innerW, maxLines - n)
            )
            addRows(chromeRows(bodyWidth, compact, fenceLang.length > 0))
          }

          inFence = false
        }
        // Non-closing lines inside a fence body contribute nothing here —
        // the wrap walk over the content slice already counted them.
      } else {
        const openMatch = line.match(FENCE_OPEN_RE)

        if (openMatch) {
          inFence = true
          fenceChar = openMatch[1]![0]
          fenceLen = openMatch[1]!.length
          fenceLang = openMatch[2]!.trim().toLowerCase()
          fenceContentStart = i + 1
        } else {
          addRows(Math.max(1, Math.ceil((lineEnd - lineStart) / w)))
        }
      }

      lineStart = i + 1

      if (n >= maxLines) {
        return maxLines
      }
    }

    i++
  }

  // Unclosed fence: the renderer treats the rest of the document as code.
  // Same md/markdown rule as the closed path.
  if (inFence) {
    if (MARKDOWN_FENCE_LANGS.has(fenceLang)) {
      addRows(wrappedLines(sliceFenceContent(fenceContentStart, lineStart), w, maxLines - n))
    } else {
      addRows(wrappedLines(sliceFenceContent(fenceContentStart, lineStart), innerW, maxLines - n))
      addRows(chromeRows(bodyWidth, compact, fenceLang.length > 0))
    }
  }

  return n
}

export const estimatedMsgHeight = (
  msg: Msg,
  cols: number,
  {
    compact,
    details,
    leadGap = false,
    thinkingVisible = details,
    toolsVisible = details,
    userPrompt = '',
    withSeparator = false
  }: {
    compact: boolean
    details: boolean
    leadGap?: boolean
    thinkingVisible?: boolean
    toolsVisible?: boolean
    userPrompt?: string
    withSeparator?: boolean
  }
) => {
  if (msg.kind === 'intro') {
    return msg.info?.version ? 9 : 5
  }

  if (msg.kind === 'panel') {
    return Math.max(3, (msg.panelData?.sections.length ?? 1) * 2 + 1)
  }

  if (msg.kind === 'trail' && msg.todos?.length) {
    if (msg.todoCollapsedByDefault) {
      return 2
    }

    return Math.max(2, msg.todos.length + 2)
  }

  const bodyWidth = transcriptBodyWidth(cols, msg.role, userPrompt, TERMUX_TUI_MODE)
  const text = msg.text
  // Fence-aware so fenced-code panels don't undercount virtual transcript
  // spacers (see PR #75783 review comment r3694319284). The walk is
  // bounded by the same MAX_ESTIMATE_LINES cap as `wrappedLines`, so a
  // multi-megabyte single line still returns in O(width) time.
  let h = estimateBodyHeight(text || ' ', bodyWidth, compact)

  if (!compact && msg.role === 'assistant') {
    // Paragraph gaps add up to 6 extra rows of breathing room. Slice
    // first so the regex never walks more than the first ~16k chars of
    // a giant assistant message — post-mount Yoga measurement converges
    // to the real height regardless of how the estimate undercounts.
    const scan = text.length > 16_000 ? text.slice(0, 16_000) : text
    h += Math.min(6, (scan.match(/\n\s*\n/g) ?? []).length)
  }

  if (details) {
    const hasVisibleTools = toolsVisible && Boolean(msg.tools?.length)
    const hasVisibleThinking = thinkingVisible && /\S/.test(msg.thinking ?? '')
    const hasVisibleDetails = hasVisibleTools || hasVisibleThinking

    if (hasVisibleDetails) {
      h +=
        (hasVisibleTools ? (msg.tools?.length ?? 0) : 0) +
        (hasVisibleThinking ? wrappedLines(msg.thinking ?? '', bodyWidth) : 0)

      if (msg.role === 'assistant' && /\S/.test(msg.text)) {
        h += 2
      }
    }
  }

  if (msg.role === 'user' || msg.kind === 'diff') {
    // Top + bottom blank line.
    h += 2
  } else if (msg.kind === 'slash') {
    h++
  }

  // Group-boundary blank line owned by BlockSlot: model prose, reasoning/tool
  // trails, and notes/errors each start a new visual group when the block
  // above them is a different kind. The caller resolves the boundary against
  // the previous row (see domain/blockLayout.ts::hasLeadGap) and passes the
  // result here so the estimate matches the rendered marginTop before Yoga
  // remeasures. user / diff / slash never set this — they own their margins.
  if (leadGap) {
    h++
  }

  // Inter-turn separator above non-first user messages (1 rule row + 1
  // top-margin row). The render-side gate is in appLayout.tsx; we trust
  // the caller to pass `withSeparator` only when it matches that gate.
  if (withSeparator) {
    h += 2
  }

  return Math.max(1, h)
}
