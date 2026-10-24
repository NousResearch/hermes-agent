import { afterEach, describe, expect, it, vi } from 'vitest'

import { $clipboardFeedback, reportClipboardCopy } from '../app/clipboardFeedbackStore.js'

afterEach(() => {
  vi.useRealTimers()
  $clipboardFeedback.set(null)
})

describe('clipboard feedback', () => {
  it('counts visible characters and expires without erasing a newer copy', () => {
    vi.useFakeTimers()
    reportClipboardCopy('🧪e\u0301')
    expect($clipboardFeedback.get()?.text).toBe('copied 2 characters')
    vi.advanceTimersByTime(1500)
    reportClipboardCopy('x')
    vi.advanceTimersByTime(1000)
    expect($clipboardFeedback.get()?.text).toBe('copied 1 character')
    vi.advanceTimersByTime(1000)
    expect($clipboardFeedback.get()).toBeNull()
  })
})
