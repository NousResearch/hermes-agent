import { describe, expect, it } from 'vitest'

import { formatAbandonedClarify, formatTokenCount, stripTrailingPasteNewlines } from './text.js'

describe('formatTokenCount', () => {
  it('matches the backend K/M rules', () => {
    expect(formatTokenCount(0)).toBe('0')
    expect(formatTokenCount(234)).toBe('234')
    expect(formatTokenCount(999)).toBe('999')
    expect(formatTokenCount(1000)).toBe('1K')
    expect(formatTokenCount(1520)).toBe('1.52K')
    expect(formatTokenCount(23500)).toBe('23.5K')
    expect(formatTokenCount(123456)).toBe('123K')
    expect(formatTokenCount(1234567)).toBe('1.23M')
    expect(formatTokenCount(12500000)).toBe('12.5M')
  })

  it('clamps negatives / non-finite to 0', () => {
    expect(formatTokenCount(-5)).toBe('0')
    expect(formatTokenCount(Number.NaN)).toBe('0')
  })
})

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

describe('formatAbandonedClarify', () => {
  it('renders the question, numbered options, and reason', () => {
    const out = formatAbandonedClarify('How do you want to scope?', ['Option A', 'Option B', 'Option C'], 'timed out')

    expect(out).toBe(
      [
        'ask How do you want to scope?',
        '  1. Option A',
        '  2. Option B',
        '  3. Option C',
        '  (timed out — no selection)'
      ].join('\n')
    )
  })

  it('handles a prompt with no choices (free-text clarify)', () => {
    const out = formatAbandonedClarify('What is the target branch?', null, 'cancelled')

    expect(out).toBe(['ask What is the target branch?', '  (cancelled — no selection)'].join('\n'))
  })

  it('trims surrounding whitespace on the question', () => {
    const out = formatAbandonedClarify('  trailing space  ', [], 'timed out')

    expect(out.split('\n')[0]).toBe('ask trailing space')
  })

  it('numbers options 1-based to match the live ClarifyPrompt', () => {
    const out = formatAbandonedClarify('q', ['first'], 'timed out')

    expect(out).toContain('  1. first')
    expect(out).not.toContain('  0.')
  })
})
