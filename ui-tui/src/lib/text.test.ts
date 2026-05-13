import { describe, expect, it } from 'vitest'

import { normalizeComposerPasteText, stripTrailingPasteNewlines } from './text.js'

describe('stripTrailingPasteNewlines', () => {
  it('removes trailing newline runs from pasted text', () => {
    expect(stripTrailingPasteNewlines('alpha\n')).toBe('alpha')
    expect(stripTrailingPasteNewlines('alpha\nbeta\n\n')).toBe('alpha\nbeta')
  })

  it('preserves interior newlines', () => {
    expect(stripTrailingPasteNewlines('alpha\nbeta\ngamma')).toBe('alpha\nbeta\ngamma')
  })

  it('preserves newline-only pastes', () => {
    expect(stripTrailingPasteNewlines('\n\n')).toBe('\n\n')
  })
})

describe('normalizeComposerPasteText', () => {
  it('leaves pasted text at or below three content lines multiline', () => {
    expect(normalizeComposerPasteText('alpha\nbeta\ngamma')).toBe('alpha\nbeta\ngamma')
    expect(normalizeComposerPasteText('alpha\nbeta\ngamma\n')).toBe('alpha\nbeta\ngamma')
  })

  it('flattens pasted text with more than three content lines into one readable line', () => {
    expect(normalizeComposerPasteText('alpha\n  beta\n\ngamma\ndelta')).toBe('alpha beta gamma delta')
    expect(normalizeComposerPasteText('alpha\r\nbeta\r\ngamma\r\ndelta')).toBe('alpha beta gamma delta')
  })
})
