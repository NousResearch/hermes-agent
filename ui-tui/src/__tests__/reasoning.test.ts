import { describe, expect, it } from 'vitest'

import { hasReasoningTag, splitReasoning } from '../lib/reasoning.js'
import { cleanThinkingText } from '../lib/text.js'

describe('splitReasoning', () => {
  it('extracts <think>…</think> and strips it from text', () => {
    const { reasoning, text } = splitReasoning('<think>plotting</think>\n\nhere is the answer')

    expect(reasoning).toBe('plotting')
    expect(text).toBe('here is the answer')
  })

  it('handles multiple tag shapes', () => {
    const input = '<reasoning>a</reasoning> <THINKING>b</THINKING> <thought>c</thought> body'
    const { reasoning, text } = splitReasoning(input)

    expect(reasoning).toContain('a')
    expect(reasoning).toContain('b')
    expect(reasoning).toContain('c')
    expect(text).toBe('body')
  })

  it('treats unclosed leading <think>… as reasoning (real reasoning-model stream)', () => {
    const { reasoning, text } = splitReasoning('<think>still deciding')

    expect(reasoning).toBe('still deciding')
    expect(text).toBe('')
  })

  it('does not strip trailing prose after a stray mid-text <think> mention', () => {
    // Regression for "TUI eats last paragraph of output": when the model
    // emits a literal `<think>` somewhere in prose (quoted explanation, code
    // example, partial stream-mid-tag), the trailing greedy unclosed-tag
    // regex used to consume every paragraph after it. Real unclosed
    // reasoning blocks always lead the message — anchor to ^ so prose
    // mentions are preserved.
    const { reasoning, text } = splitReasoning(
      'final answer paragraph one.\n\n<think>internal note never closed\n\nfinal answer paragraph two.'
    )

    expect(reasoning).toBe('')
    expect(text).toBe('final answer paragraph one.\n\n<think>internal note never closed\n\nfinal answer paragraph two.')
  })

  it('returns empty reasoning and untouched text when no tags present', () => {
    const { reasoning, text } = splitReasoning('plain body with no tags')

    expect(reasoning).toBe('')
    expect(text).toBe('plain body with no tags')
  })

  it('preserves text when reasoning block is empty', () => {
    const { reasoning, text } = splitReasoning('<think></think>only body')

    expect(reasoning).toBe('')
    expect(text).toBe('only body')
  })

  it('detects presence of any supported tag', () => {
    expect(hasReasoningTag('pre <think>x</think> post')).toBe(true)
    expect(hasReasoningTag('pre <reasoning>x</reasoning>')).toBe(true)
    expect(hasReasoningTag('<REASONING_SCRATCHPAD>x</REASONING_SCRATCHPAD>')).toBe(true)
    expect(hasReasoningTag('no tags at all')).toBe(false)
  })
})

describe('cleanThinkingText', () => {
  it('removes face/status ticker fragments while preserving real reasoning', () => {
    expect(
      cleanThinkingText(
        '(¬_¬) synthesizing...**Resolving comments on GitHub**\n( ͡° ͜ʖ ͡°) musing...\nActual step\n٩(๑❛ᴗ❛๑)۶ contemplating...next step'
      )
    ).toBe('**Resolving comments on GitHub**\nActual step\nnext step')
  })
})
