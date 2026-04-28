import { describe, expect, it } from 'vitest'

import { stickyPromptFromViewport } from '../domain/viewport.js'

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
})
