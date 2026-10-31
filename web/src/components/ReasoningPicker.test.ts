import { describe, expect, it } from 'vitest'

import { reasoningPickerOptions } from './ReasoningPicker'

describe('reasoningPickerOptions', () => {
  it('keeps an existing saved effort visible when a declared vocabulary no longer includes it', () => {
    expect(reasoningPickerOptions('max', ['low', 'high']).map(option => option.value)).toEqual(['low', 'high', 'max'])
    expect(reasoningPickerOptions('none', ['low', 'high']).map(option => option.value)).toEqual(['low', 'high', 'none'])
  })

  it('does not offer a control for a provider declaration that accepts no reasoning parameter', () => {
    expect(reasoningPickerOptions('medium', [])).toEqual([])
  })
})
