import { describe, expect, it } from 'vitest'

import { hasReasoningTag, splitReasoning } from '../lib/reasoning.js'

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

  it('treats unclosed trailing <think>… as reasoning', () => {
    const { reasoning, text } = splitReasoning('answer start <think>still deciding')

    expect(reasoning).toBe('still deciding')
    expect(text).toBe('answer start')
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
