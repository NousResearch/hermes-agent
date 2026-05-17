import { describe, expect, it } from 'vitest'

import { promptAnchorFromViewport, promptJumpStateFromViewport, stickyPromptFromViewport } from '../domain/viewport.js'

describe('promptAnchorFromViewport', () => {
  const messages = [
    { role: 'system' as const, text: '' },
    { role: 'user' as const, text: 'first prompt' },
    { role: 'assistant' as const, text: 'first answer' },
    { role: 'user' as const, text: 'second prompt' },
    { role: 'assistant' as const, text: 'second answer' },
    { role: 'user' as const, text: 'third prompt' }
  ]

  const offsets = [0, 1, 4, 20, 23, 40, 44]

  it('snaps backward to the nearest user prompt above the viewport', () => {
    expect(promptAnchorFromViewport(messages, offsets, 34, 'previous')).toEqual({ index: 3, top: 20 })
  })

  it('snaps forward to the next user prompt below the viewport', () => {
    expect(promptAnchorFromViewport(messages, offsets, 20, 'next')).toEqual({ index: 5, top: 40 })
  })

  it('skips the current prompt when already parked on its boundary', () => {
    expect(promptAnchorFromViewport(messages, offsets, 20, 'previous')).toEqual({ index: 1, top: 1 })
  })

  it('returns null at the transcript edges', () => {
    expect(promptAnchorFromViewport(messages, offsets, 1, 'previous')).toBeNull()
    expect(promptAnchorFromViewport(messages, offsets, 44, 'next')).toBeNull()
  })
})

describe('promptJumpStateFromViewport', () => {
  const messages = [
    { role: 'system' as const, text: '' },
    { role: 'user' as const, text: 'first prompt' },
    { role: 'assistant' as const, text: 'first answer' },
    { role: 'user' as const, text: 'second prompt' },
    { role: 'assistant' as const, text: 'second answer' },
    { role: 'user' as const, text: 'third prompt' },
    { role: 'assistant' as const, text: 'third answer' }
  ]

  const offsets = [0, 1, 4, 20, 23, 40, 44, 80]

  it('hides all jump controls at live bottom', () => {
    expect(promptJumpStateFromViewport(messages, offsets, 70, true)).toEqual({
      bottomMode: 'hidden',
      hasNextPrompt: false,
      hasPreviousPrompt: false
    })
  })

  it('shows previous prompt plus centered bottom when only live bottom is below', () => {
    expect(promptJumpStateFromViewport(messages, offsets, 55, false)).toEqual({
      bottomMode: 'center',
      hasNextPrompt: false,
      hasPreviousPrompt: true
    })
  })

  it('keeps two banners while the viewport is still between bottom and the latest prompt', () => {
    expect(promptJumpStateFromViewport(messages, offsets, 42, false, 52)).toEqual({
      bottomMode: 'center',
      hasNextPrompt: false,
      hasPreviousPrompt: true
    })
  })

  it('keeps two banners while the latest prompt is still visible in the viewport', () => {
    expect(promptJumpStateFromViewport(messages, offsets, 35, false, 45)).toEqual({
      bottomMode: 'center',
      hasNextPrompt: false,
      hasPreviousPrompt: true
    })
  })

  it('shows previous, next, and left bottom once scrolled higher than the latest prompt', () => {
    expect(promptJumpStateFromViewport(messages, offsets, 29, false, 39)).toEqual({
      bottomMode: 'left',
      hasNextPrompt: true,
      hasPreviousPrompt: true
    })
  })
})

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
