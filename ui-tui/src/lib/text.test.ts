import { describe, expect, it } from 'vitest'

import { t } from '../i18n/runtime.js'

import {
  clarifyBatchRevisitState,
  formatAbandonedClarify,
  formatAbandonedClarifyBatch,
  stripTrailingPasteNewlines
} from './text.js'

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
        t('libText.text.clarifyHead', 'How do you want to scope?'),
        '  1. Option A',
        '  2. Option B',
        '  3. Option C',
        `  ${t('libText.text.clarifyNoSelection', 'timed out')}`
      ].join('\n')
    )
  })

  it('handles a prompt with no choices (free-text clarify)', () => {
    const out = formatAbandonedClarify('What is the target branch?', null, 'cancelled')

    expect(out).toBe(
      [
        t('libText.text.clarifyHead', 'What is the target branch?'),
        `  ${t('libText.text.clarifyNoSelection', 'cancelled')}`
      ].join('\n')
    )
  })

  it('trims surrounding whitespace on the question', () => {
    const out = formatAbandonedClarify('  trailing space  ', [], 'timed out')

    expect(out.split('\n')[0]).toBe(t('libText.text.clarifyHead', 'trailing space'))
  })
})

describe('formatAbandonedClarifyBatch', () => {
  it('shows locked answers and marks unanswered questions', () => {
    const out = formatAbandonedClarifyBatch(
      [
        { qid: 'q0', question: 'One?' },
        { qid: 'q1', question: 'Two?' }
      ],
      { q0: 'alpha' },
      'timed out'
    )

    expect(out).toBe(
      [
        t('libText.text.clarifyBatchHead', 2),
        `  ${t('libText.text.clarifyAnswered', 'One?', 'alpha')}`,
        `  ${t('libText.text.clarifyUnanswered', 'Two?')}`,
        `  ${t('libText.text.clarifyBatchReason', 'timed out')}`
      ].join('\n')
    )
  })

  it('treats an empty locked answer as unanswered in the record', () => {
    const out = formatAbandonedClarifyBatch([{ qid: 'q0', question: 'One?' }], { q0: '' }, 'cancelled')

    expect(out).toContain(t('libText.text.clarifyUnanswered', 'One?'))
  })
})

describe('clarifyBatchRevisitState', () => {
  it('restores the cursor onto a choice answer', () => {
    expect(clarifyBatchRevisitState(['red', 'blue'], 'blue')).toEqual({ custom: '', sel: 1 })
  })

  it('stages a typed answer on the Other row for editing', () => {
    expect(clarifyBatchRevisitState(['red', 'blue'], 'chartreuse')).toEqual({ custom: 'chartreuse', sel: 2 })
  })

  it('stages a typed answer for an open-ended question (no choices)', () => {
    expect(clarifyBatchRevisitState([], 'free text')).toEqual({ custom: 'free text', sel: 0 })
  })

  it('resets cleanly for unanswered and empty answers', () => {
    expect(clarifyBatchRevisitState(['red'], undefined)).toEqual({ custom: '', sel: 0 })
    expect(clarifyBatchRevisitState(['red'], '')).toEqual({ custom: '', sel: 0 })
  })
})
