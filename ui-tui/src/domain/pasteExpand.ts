import type { ComposerToken } from '../app/interfaces.js'
import { pasteTokenLabel } from '../lib/text.js'

/**
 * A collapsed paste the user expanded back into the composer: `[start, end)`
 * is the pasted text, now ordinary editable characters. Collapsing turns
 * whatever sits in that range (edits included) back into a `[[ … ]]` chip.
 */
export interface PasteSpan {
  end: number
  start: number
}

/**
 * Keep `span` on the same characters across an edit `prev` → `next`.
 *
 * The edit is the region between the common prefix and the common suffix.
 * Typing inside the span, or at its very end (where the cursor sits right
 * after expanding), grows it; typing before it shifts it. An edit that
 * replaces the whole span (clear, history recall) ends it.
 */
export function shiftSpan(prev: string, next: string, span: PasteSpan): null | PasteSpan {
  if (prev === next) {
    return span
  }

  const max = Math.min(prev.length, next.length)
  let head = 0

  while (head < max && prev[head] === next[head]) {
    head++
  }

  let tail = 0

  while (tail < max - head && prev[prev.length - 1 - tail] === next[next.length - 1 - tail]) {
    tail++
  }

  const oldEnd = prev.length - tail
  const newEnd = next.length - tail
  const delta = newEnd - oldEnd
  let { end, start } = span

  if (oldEnd <= start) {
    start += delta
    end += delta
  } else if (head > end) {
    // after the span: nothing moves
  } else if (head >= start && oldEnd <= end) {
    end += delta
  } else if (head <= start && oldEnd >= end) {
    return null
  } else if (head < start) {
    start = head
    end += delta
  } else {
    end = newEnd
  }

  return end > start ? { end, start } : null
}

/**
 * Every token's `[[ … ]]` chip in `value`, in text order. Same-label tokens
 * claim successive occurrences in list order, the order `expandTokens` reads.
 */
const tokenChips = (value: string, tokens: ComposerToken[]) => {
  const seen = new Map<string, number>()
  const chips: { at: number; token: ComposerToken }[] = []

  for (const token of tokens) {
    const from = seen.get(token.label) ?? 0
    const at = value.indexOf(token.label, from)

    if (at < 0) {
      continue
    }

    seen.set(token.label, at + token.label.length)
    chips.push({ at, token })
  }

  return chips.sort((a, b) => a.at - b.at)
}

const pasteChips = (value: string, tokens: ComposerToken[]) =>
  tokenChips(value, tokens).filter(c => c.token.kind === 'paste')

export const hasPasteChip = (value: string, tokens: ComposerToken[]) => pasteChips(value, tokens).length > 0

/**
 * Expand the paste chip under the cursor (or the last one) into its full text.
 * The token leaves the token list: its text is now the composer's own.
 */
export function expandPasteChip(value: string, cursor: number, tokens: ComposerToken[]) {
  const chips = pasteChips(value, tokens)
  const chip = chips.find(c => cursor >= c.at && cursor <= c.at + c.token.label.length) ?? chips.at(-1)

  if (!chip) {
    return null
  }

  const text = chip.token.text ?? ''

  return {
    cursor: chip.at + text.length,
    span: { end: chip.at + text.length, start: chip.at },
    tokens: tokens.filter(t => t !== chip.token),
    value: value.slice(0, chip.at) + text + value.slice(chip.at + chip.token.label.length)
  }
}

/** Whether `cursor` sits inside the expanded span, where typed or pasted text joins the paste. */
export const insideSpan = (span: null | PasteSpan, cursor: number) =>
  Boolean(span && cursor > span.start && cursor <= span.end)

/**
 * Fold the expanded span back into a chip whose payload is the span's current
 * text. The new token takes its place among same-label tokens by position, so
 * `expandTokens` still pairs every chip with its own body. A span that holds
 * another live chip (an image attached at its end) is not folded (null): the
 * outer payload would carry that chip's label, not its content.
 */
export function collapsePasteSpan(value: string, span: PasteSpan, tokens: ComposerToken[]) {
  if (tokenChips(value, tokens).some(c => c.at < span.end && c.at + c.token.label.length > span.start)) {
    return null
  }

  const text = value.slice(span.start, span.end)
  const label = pasteTokenLabel(text, text.split('\n').length)
  const token: ComposerToken = { kind: 'paste', label, text }
  const before = value.slice(0, span.start).split(label).length - 1
  let seen = 0
  let at = tokens.length

  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i]!.label === label && seen++ === before) {
      at = i

      break
    }
  }

  return {
    cursor: span.start + label.length,
    tokens: [...tokens.slice(0, at), token, ...tokens.slice(at)],
    value: value.slice(0, span.start) + label + value.slice(span.end)
  }
}
