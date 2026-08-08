const TAGS = ['think', 'reasoning', 'thinking', 'thought', 'REASONING_SCRATCHPAD'] as const

export interface SplitReasoning {
  reasoning: string
  text: string
}

// Assistant-message keys the gateway forwards verbatim out of persisted
// history. Keep in sync with `reasoning_keys` in `_history_to_messages`
// (tui_gateway/server.py) — that tuple is the wire contract, and the two
// structured members are provider replay payloads, not plain strings.
export interface ReasoningFields {
  codex_reasoning_items?: unknown
  reasoning?: unknown
  reasoning_content?: unknown
  reasoning_details?: unknown
}

// Human-readable fields on a structured reasoning entry, in the precedence
// `extract_reasoning` (agent/agent_runtime_helpers.py) already uses:
//   * Anthropic thinking block  → {type: 'thinking', thinking, signature}
//   * OpenRouter summary/text   → {type: 'reasoning.summary', summary} etc.
//   * Codex Responses item      → {type: 'reasoning', summary: [{text}], …}
// `signature`, `data` and `encrypted_content` are deliberately absent: those
// are opaque replay blobs (redacted_thinking, reasoning.encrypted_content,
// Codex encrypted reasoning) that would render as base64 noise, so an entry
// carrying only those contributes nothing to the displayed text.
const DETAIL_TEXT_FIELDS = ['summary', 'thinking', 'content', 'text'] as const

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

// A readable field is either prose or a list of `{text}` parts (Codex stores
// `summary` as `[{type: 'summary_text', text}]`, OpenRouter as a bare string).
const readableText = (value: unknown): string => {
  if (typeof value === 'string') {
    return value
  }

  if (Array.isArray(value)) {
    return value
      .map(part => (isRecord(part) && typeof part.text === 'string' ? part.text : ''))
      .filter(Boolean)
      .join('\n')
  }

  return ''
}

/**
 * Flatten a persisted assistant row's reasoning fields into the single
 * `Msg.thinking` string the transcript renders.
 *
 * Mirrors `extract_reasoning`: prose fields first, then each structured
 * entry's readable member, de-duplicated and joined with a blank line. The
 * de-duplication is load-bearing rather than cosmetic — the Anthropic
 * transport populates `reasoning` *and* `reasoning_details` from the same
 * thinking blocks, so a resumed Claude turn would otherwise show its
 * reasoning twice.
 */
export const flattenReasoning = (row: ReasoningFields): string => {
  const parts: string[] = []

  const add = (value: unknown) => {
    const text = readableText(value).trim()

    if (text && !parts.includes(text)) {
      parts.push(text)
    }
  }

  add(row.reasoning)
  add(row.reasoning_content)

  // `reasoning_details` before `codex_reasoning_items`, matching the order the
  // gateway walks its `reasoning_keys` tuple.
  for (const entries of [row.reasoning_details, row.codex_reasoning_items]) {
    if (!Array.isArray(entries)) {
      continue
    }

    for (const entry of entries) {
      if (!isRecord(entry)) {
        continue
      }

      // First readable member wins — an entry never carries two of these.
      for (const name of DETAIL_TEXT_FIELDS) {
        if (readableText(entry[name]).trim()) {
          add(entry[name])

          break
        }
      }
    }
  }

  return parts.join('\n\n')
}

export function splitReasoning(input: string): SplitReasoning {
  let text = input
  const reasoning: string[] = []

  for (const tag of TAGS) {
    const paired = new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>\\s*`, 'gi')
    text = text.replace(paired, (_m, inner: string) => {
      const trimmed = inner.trim()

      if (trimmed) {
        reasoning.push(trimmed)
      }

      return ''
    })

    // Anchor to start-of-input so a literal `<think>` mid-prose (model quoting
    // the word, code blocks containing the tag, etc.) doesn't eat every
    // paragraph after it. Real unclosed reasoning blocks always lead the
    // message — that's how reasoning models stream. See test
    // "does not strip trailing prose after a stray mid-text <think> mention".
    const unclosed = new RegExp(`^\\s*<${tag}>([\\s\\S]*)$`, 'i')
    text = text.replace(unclosed, (_m, inner: string) => {
      const trimmed = inner.trim()

      if (trimmed) {
        reasoning.push(trimmed)
      }

      return ''
    })
  }

  return {
    reasoning: reasoning.join('\n\n').trim(),
    text: text.trim()
  }
}

export const hasReasoningTag = (input: string) => {
  for (const tag of TAGS) {
    if (input.includes(`<${tag}>`)) {
      return true
    }
  }

  return false
}
