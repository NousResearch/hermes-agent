import { describe, expect, it } from 'vitest'

import { t } from '../i18n/runtime.js'

import { clarifyRevisitState, formatAbandonedClarify, stripTrailingPasteNewlines } from './text.js'

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
  it('shows locked answers and marks unanswered questions', () => {
    const out = formatAbandonedClarify(
      [
        { qid: 'q0', question: 'One?' },
        { qid: 'q1', question: 'Two?' }
      ],
      { q0: 'alpha' },
      'timed out'
    )

    expect(out).toBe(
      [
        t('libText.text.clarifyHead', 2),
        `  ${t('libText.text.clarifyAnswered', 'One?', 'alpha')}`,
        `  ${t('libText.text.clarifyUnanswered', 'Two?')}`,
        `  ${t('libText.text.clarifyReason', 'timed out')}`
      ].join('\n')
    )
  })

  it('treats an empty locked answer as unanswered in the record', () => {
    const out = formatAbandonedClarify([{ qid: 'q0', question: 'One?' }], { q0: '' }, 'cancelled')

    expect(out).toContain(t('libText.text.clarifyUnanswered', 'One?'))
  })
})

describe('clarifyRevisitState', () => {
  it('restores the cursor onto a choice answer', () => {
    expect(clarifyRevisitState(['red', 'blue'], 'blue')).toEqual({ custom: '', picked: [], sel: 1 })
  })

  it('stages a typed answer on the Other row for editing', () => {
    expect(clarifyRevisitState(['red', 'blue'], 'chartreuse')).toEqual({ custom: 'chartreuse', picked: [], sel: 2 })
  })

  it('stages a typed answer for an open-ended question (no choices)', () => {
    expect(clarifyRevisitState([], 'free text')).toEqual({ custom: 'free text', picked: [], sel: 0 })
  })

  it('resets cleanly for unanswered and empty answers', () => {
    expect(clarifyRevisitState(['red'], undefined)).toEqual({ custom: '', picked: [], sel: 0 })
    expect(clarifyRevisitState(['red'], '')).toEqual({ custom: '', picked: [], sel: 0 })
  })
})
