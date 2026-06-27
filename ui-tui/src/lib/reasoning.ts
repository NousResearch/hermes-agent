const TAGS = ['think', 'reasoning', 'thinking', 'thought', 'REASONING_SCRATCHPAD'] as const

export interface SplitReasoning {
  reasoning: string
  text: string
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

    const unclosed = new RegExp(`<${tag}>([\\s\\S]*)$`, 'i')
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
