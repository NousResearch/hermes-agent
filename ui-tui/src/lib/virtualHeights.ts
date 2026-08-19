import { TERMUX_TUI_MODE } from '../config/env.js'
import type { Msg } from '../types.js'

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

const FENCE_OPEN_RE = /^\s*(`{3,}|~{3,})(.*)$/
const FENCE_CLOSE_RE = /^\s*(`{3,}|~{3,})\s*$/
const MARKDOWN_FENCE_LANGS = new Set(['md', 'markdown'])

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

export const fencedWrappedLines = (
  text: string,
  width: number,
  compact: boolean,
  maxLines: number = MAX_ESTIMATE_LINES
) => {
  const bodyWidth = Math.max(1, width - 2)
  const frameRows = compact || width < 20 ? 1 : 2
  const lines = text.split('\n')
  let rows = 0
  let fenceChar = ''
  let fenceLength = 0
  let fenceLanguage = ''
  let fenceBody: string[] = []

  const addRows = (count: number) => {
    rows = Math.min(maxLines, rows + count)
  }

  const finishFence = () => {
    if (MARKDOWN_FENCE_LANGS.has(fenceLanguage)) {
      addRows(fenceBody.length ? wrappedLines(fenceBody.join('\n'), width, maxLines - rows) : 0)
    } else {
      addRows(fenceBody.length ? wrappedLines(fenceBody.join('\n'), bodyWidth, maxLines - rows) : 0)
      addRows(frameRows)
    }

    fenceChar = ''
    fenceLength = 0
    fenceLanguage = ''
    fenceBody = []
  }

  for (const line of lines) {
    if (!fenceChar) {
      const open = line.match(FENCE_OPEN_RE)

      if (open) {
        fenceChar = open[1]![0]
        fenceLength = open[1]!.length
        fenceLanguage = open[2]!.trim().toLowerCase()
        fenceBody = []
      } else {
        addRows(wrappedLines(line, width, maxLines - rows))
      }
    } else {
      const close = line.match(FENCE_CLOSE_RE)?.[1]

      if (close && close[0] === fenceChar && close.length >= fenceLength) {
        finishFence()
      } else {
        fenceBody.push(line)
      }
    }

    if (rows >= maxLines) {
      return maxLines
    }
  }

  if (fenceChar) {
    finishFence()
  }

  return rows
}

export const estimatedMsgHeight = (
  msg: Msg,
  cols: number,
  {
    compact,
    details,
    leadGap = false,
    thinkingVisible = details,
    thinkingExpanded = thinkingVisible,
    toolsVisible = details,
    userPrompt = '',
    withSeparator = false
  }: {
    compact: boolean
    details: boolean
    leadGap?: boolean
    thinkingExpanded?: boolean
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
  let h = fencedWrappedLines(text || ' ', bodyWidth, compact)

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
        (hasVisibleThinking ? (thinkingExpanded ? wrappedLines(msg.thinking ?? '', bodyWidth) : 1) : 0)

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
