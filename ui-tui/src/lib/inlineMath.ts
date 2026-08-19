export interface InlineMathSpan {
  content: string
  index: number
  raw: string
}

const isEscapedAt = (text: string, index: number) => {
  let slashCount = 0

  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) {
    slashCount += 1
  }

  return slashCount % 2 === 1
}

const isUnescapedDollarAt = (text: string, index: number) => text[index] === '$' && !isEscapedAt(text, index)

export const unescapeMarkdownDollars = (text: string) =>
  text.replace(/(\\+)\$/g, (_match, slashes: string) => `${'\\'.repeat(Math.floor(slashes.length / 2))}$`)

const isLikelyNumericInlineMath = (body: string, followingCharacter: string) => {
  const value = body.trim()

  if (!/^\d/u.test(value)) {
    return true
  }

  // A compact price range such as `$5-$10` presents its second price opener
  // as the first candidate closer. A trailing operator is therefore evidence
  // that this is currency prose, not a balanced numeric formula.
  if (/[+\-*/=<>^_,;:(]$/u.test(value)) {
    return false
  }

  if (/https?:\/\//iu.test(value)) {
    return false
  }

  // `$5$10` and `$5$x$` are more likely adjacent currency/prose openers.
  // Preserve the numeric candidate only when its body carries a math signal.
  if (/^\p{N}/u.test(followingCharacter)) {
    return false
  }

  if (/^[\p{L}\\]/u.test(followingCharacter)) {
    return /\\[A-Za-z]+|[+*/=<>^_{}]/u.test(value)
  }

  return true
}

const findNextUnescapedDollar = (text: string, fromIndex: number) => {
  for (let cursor = fromIndex; cursor < text.length && text[cursor] !== '\n'; cursor += 1) {
    if (text[cursor] === '$' && !isEscapedAt(text, cursor)) {
      return cursor
    }
  }

  return -1
}

/**
 * Find the next single-dollar inline-math span.
 *
 * Delimiter recognition is deliberately a scanner rather than another
 * lookaround-heavy regex: whether `$` is escaped depends on the parity of the
 * entire preceding backslash run, and an escaped dollar inside a formula must
 * be skipped while looking for its real closer.
 */
export const findNextInlineMath = (text: string, fromIndex = 0): InlineMathSpan | null => {
  for (let openingIndex = fromIndex; openingIndex < text.length; openingIndex += 1) {
    if (
      !isUnescapedDollarAt(text, openingIndex) ||
      isUnescapedDollarAt(text, openingIndex - 1) ||
      isUnescapedDollarAt(text, openingIndex + 1)
    ) {
      continue
    }

    const closingIndex = findNextUnescapedDollar(text, openingIndex + 1)

    if (closingIndex < 0) {
      // Inline math cannot cross a newline, but an unmatched opener on one
      // line must not hide valid math on a later line.
      openingIndex = text.indexOf('\n', openingIndex)

      if (openingIndex < 0) {
        return null
      }

      continue
    }

    const body = text.slice(openingIndex + 1, closingIndex)

    if (
      isUnescapedDollarAt(text, closingIndex - 1) ||
      isUnescapedDollarAt(text, closingIndex + 1) ||
      !body ||
      /^\s/u.test(body) ||
      /\s$/u.test(body) ||
      !isLikelyNumericInlineMath(body, text[closingIndex + 1] || '')
    ) {
      continue
    }

    return {
      content: body,
      index: openingIndex,
      raw: text.slice(openingIndex, closingIndex + 1)
    }
  }

  return null
}

export function* inlineMathSpans(text: string): Generator<InlineMathSpan> {
  let cursor = 0

  while (cursor < text.length) {
    const span = findNextInlineMath(text, cursor)

    if (!span) {
      return
    }

    yield span
    cursor = span.index + span.raw.length
  }
}

export const stripInlineMathDelimiters = (text: string) => {
  let out = ''
  let cursor = 0

  for (const span of inlineMathSpans(text)) {
    out += text.slice(cursor, span.index)
    out += span.content
    cursor = span.index + span.raw.length
  }

  return out + text.slice(cursor)
}
