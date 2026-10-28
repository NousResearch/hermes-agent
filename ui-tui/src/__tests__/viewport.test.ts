import { describe, expect, it } from 'vitest'

import { stickyPromptFromViewport, turnAnchorFromViewport } from '../domain/viewport.js'

describe('stickyPromptFromViewport', () => {
  it('hides the sticky prompt when a newer user message is already visible', () => {
    const messages = [
      { role: 'user' as const, text: 'older prompt' },
      { role: 'assistant' as const, text: 'older answer' },
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    const offsets = [0, 2, 10, 12, 20]

    expect(stickyPromptFromViewport(messages, offsets, 8, 16, false)).toBe('')
  })

  it('shows the latest user message above the viewport when no user message is visible', () => {
    const messages = [
      { role: 'user' as const, text: 'older prompt' },
      { role: 'assistant' as const, text: 'older answer' },
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    const offsets = [0, 2, 10, 12, 20]

    expect(stickyPromptFromViewport(messages, offsets, 16, 20, false)).toBe('current prompt')
  })

  it('shows the last prompt once the viewport starts after the history tail', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'completed answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 5], 8, 14, false)).toBe('current prompt')
  })

  it('shows a prompt as soon as its full row is above the viewport', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 10], 2, 8, false)).toBe('current prompt')
  })

  it('hides the sticky prompt at the bottom', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 10], 8, 10, true)).toBe('')
  })
})

describe('turnAnchorFromViewport', () => {
  it('returns null when the transcript is empty', () => {
    expect(turnAnchorFromViewport([], [], 0, 10, true, false)).toBeNull()
  })

  it('returns null when there are no user messages at all', () => {
    const messages = [
      { role: 'assistant' as const, text: 'hello' },
      { role: 'assistant' as const, text: 'world' }
    ]

    expect(turnAnchorFromViewport(messages, [0, 2], 0, 4, false, false)).toBeNull()
  })

  it('returns null when sticky to bottom and not streaming', () => {
    const messages = [
      { role: 'user' as const, text: 'prompt' },
      { role: 'assistant' as const, text: 'answer' }
    ]

    expect(turnAnchorFromViewport(messages, [0, 2, 10], 8, 10, true, false)).toBeNull()
  })

  it('live mode: streaming turn whose user message block ended above the viewport top anchors that message', () => {
    const messages = [
      { role: 'user' as const, text: 'old prompt' },
      { role: 'assistant' as const, text: 'old answer' },
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'streaming answer...' }
    ]

    const anchor = turnAnchorFromViewport(messages, [0, 2, 10, 12], 16, 22, true, true)

    expect(anchor?.mode).toBe('live')
    expect(anchor?.turnIndex).toBe(2)
    expect(anchor?.msg.text).toBe('current prompt')
  })

  it('history mode: scrolled up mid-history, nearest user message above top opens the visible turn', () => {
    const messages = [
      { role: 'user' as const, text: 'first prompt' },
      { role: 'assistant' as const, text: 'first answer' },
      { role: 'user' as const, text: 'second prompt' },
      { role: 'assistant' as const, text: 'second answer' }
    ]

    const anchor = turnAnchorFromViewport(messages, [0, 2, 6, 8], 10, 16, false, false)

    expect(anchor?.mode).toBe('history')
    expect(anchor?.turnIndex).toBe(2)
    expect(anchor?.msg.text).toBe('second prompt')
  })

  it('consecutive user messages are separate turns — anchor is the nearest user message above the viewport, not an older one', () => {
    const messages = [
      { role: 'user' as const, text: 'first' },
      { role: 'user' as const, text: 'second' },
      { role: 'assistant' as const, text: 'answer to both' }
    ]

    const anchor = turnAnchorFromViewport(messages, [0, 2, 4, 8], 6, 12, false, false)

    expect(anchor?.turnIndex).toBe(1)
    expect(anchor?.msg.text).toBe('second')
  })

  it('returns null when a user message is visible inside the viewport', () => {
    const messages = [
      { role: 'user' as const, text: 'prompt' },
      { role: 'assistant' as const, text: 'answer' }
    ]

    expect(turnAnchorFromViewport(messages, [0, 2, 10], 0, 5, false, false)).toBeNull()
  })

  it('carries the full message; truncation is the bars job', () => {
    const long = 'line one\nline two\nline three'

    const messages = [
      { role: 'user' as const, text: long },
      { role: 'assistant' as const, text: 'answer' }
    ]

    const anchor = turnAnchorFromViewport(messages, [0, 6, 12], 8, 14, false, false)

    expect(anchor?.msg.text).toBe(long)
  })
})
