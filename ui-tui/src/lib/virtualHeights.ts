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

// East-Asian "wide" code points (CJK ideographs, hiragana/katakana, hangul,
// fullwidth forms, vertical forms, CJK compatibility forms, CJK extension
// planes, ...) render at 2 terminal columns instead of 1. Counting them as
// width 1 (plain code-unit count) undercounts wrapped rows by ~2x for
// CJK-heavy text, which under-reserves virtualized transcript space —
// visible as clipped trailing lines and a scrollHeight that falls short of
// the real rendered bottom.
//
// This is a cheap code-point-range check aligned with Ink's bundled
// isEastAsianWide (packages/hermes-ink/src/ink/termio/parser.ts). It is
// NOT full grapheme/emoji-width handling (that's `stringWidth` in
// inputMetrics.ts, too costly to call per char across a multi-thousand-char
// wrap walk) so it keeps the single-pass O(text) budget this estimator
// relies on.
const isWideCodePoint = (code: number) =>
  (code >= 0x1100 && code <= 0x115f) || // Hangul Jamo
  (code >= 0x2e80 && code <= 0x9fff) || // CJK Radicals .. CJK Unified Ideographs
  (code >= 0xac00 && code <= 0xd7a3) || // Hangul Syllables
  (code >= 0xf900 && code <= 0xfaff) || // CJK Compatibility Ideographs
  (code >= 0xfe10 && code <= 0xfe1f) || // Vertical Forms
  (code >= 0xfe30 && code <= 0xfe6f) || // CJK Compatibility Forms
  (code >= 0xff00 && code <= 0xff60) || // Fullwidth Forms
  (code >= 0xffe0 && code <= 0xffe6) || // Fullwidth Signs
  (code >= 0x20000 && code <= 0x2fffd) || // CJK Extension B..F
  (code >= 0x30000 && code <= 0x3fffd) // CJK Extension G..H

const charWidth = (code: number) => (isWideCodePoint(code) ? 2 : 1)

export const wrappedLines = (text: string, width: number, maxLines: number = MAX_ESTIMATE_LINES) => {
  const w = Math.max(1, width)
  // Worst case: every cell is its own row at width=1, plus a small
  // slack for the trailing partial line. Walking past this byte budget
  // cannot increase n any further once n is already past maxLines, so
  // bail. Saves O(text) walks on multi-megabyte single-line messages.
  // (Wide chars only make rows fill faster, so this char-count budget
  // stays a safe, if generous, upper bound once widths are counted.)
  const budget = Math.min(text.length, maxLines * w + maxLines)
  let n = 0
  let lineWidth = 0

  for (let i = 0; i <= budget; i++) {
    if (i === text.length || i === budget || text.charCodeAt(i) === 10) {
      const rows = Math.max(1, Math.ceil(lineWidth / w))
      n += rows >= maxLines - n ? maxLines - n : rows
      lineWidth = 0

      if (n >= maxLines) {
        return maxLines
      }

      continue
    }

    const cc = text.charCodeAt(i)
    let code: number

    if (cc >= 0xd800 && cc <= 0xdbff) {
      // Supplementary plane character (surrogate pair). Decode the full
      // code point so isWideCodePoint can match Ink's supplementary-plane
      // ranges (U+20000–U+2FFFD, U+30000–U+3FFFD). Only consume the low
      // surrogate when it is inside this bounded walk; otherwise the next
      // iteration must still finalize the accumulated row at i===budget.
      const low = text.charCodeAt(i + 1)

      if (i + 1 < budget && low >= 0xdc00 && low <= 0xdfff) {
        code = (cc - 0xd800) * 0x400 + (low - 0xdc00) + 0x10000
        i++ // skip the low surrogate
      } else {
        code = cc // unpaired or budget-split surrogate; count this code unit once
      }
    } else {
      code = cc
    }

    lineWidth += charWidth(code)
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
  let h = wrappedLines(text || ' ', bodyWidth)

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
