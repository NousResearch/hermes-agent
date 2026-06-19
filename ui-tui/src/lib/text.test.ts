import { describe, expect, it } from 'vitest'

import { formatAbandonedClarify, stripTrailingPasteNewlines } from './text.js'

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
      ['ask How do you want to scope?', '  1. Option A', '  2. Option B', '  3. Option C', '  (timed out — no selection)'].join(
        '\n'
      )
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
