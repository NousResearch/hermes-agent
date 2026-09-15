import type { ModelOptionProvider } from '@hermes/shared/gateway-events'
import { describe, expect, it } from 'vitest'

import { draftModelNameFromArg } from '../components/activeSessionSwitcher.js'
import { modelPickerCommand, pickerOffersReasoning, REASONING_PICKER_ROWS } from '../components/modelPicker.js'

const provider = (capabilities?: ModelOptionProvider['capabilities']): ModelOptionProvider => ({
  capabilities,
  name: 'Nous Portal',
  slug: 'nous'
})

describe('ModelPicker reasoning step', () => {
  it('emits one /model request carrying provider, effort and scope', () => {
    expect(modelPickerCommand('gpt-5.6', 'nous', false, 'high')).toBe(
      'gpt-5.6 --provider nous --reasoning high --tui-session'
    )
    expect(modelPickerCommand('gpt-5.6', 'nous', true, 'none')).toBe(
      'gpt-5.6 --provider nous --reasoning none --global'
    )
    // "Keep current effort" (empty value) adds no flag at all.
    expect(modelPickerCommand('gpt-5.6', 'nous', false, '')).toBe('gpt-5.6 --provider nous --tui-session')
    expect(REASONING_PICKER_ROWS.at(-1)?.value).toBe('')
    // The new-session draft label strips the effort flag like it strips --provider.
    expect(draftModelNameFromArg(modelPickerCommand('gpt-5.6', 'nous', false, 'low'))).toBe('gpt-5.6')
  })

  it('skips the step only when the catalog says the route has no reasoning control', () => {
    expect(pickerOffersReasoning(provider({ 'gpt-5.6': { fast: false, reasoning: false } }), 'gpt-5.6')).toBe(false)
    expect(pickerOffersReasoning(provider({ 'gpt-5.6': { fast: false, reasoning: true } }), 'gpt-5.6')).toBe(true)
    expect(pickerOffersReasoning(provider(undefined), 'gpt-5.6')).toBe(true)
    expect(pickerOffersReasoning(undefined, 'gpt-5.6')).toBe(true)
  })

  it('hop Enter uses the same skip/emit as the wizard model step', () => {
    const nous = provider({ 'hermes-4': { fast: false, reasoning: false } })

    expect(pickerOffersReasoning(nous, 'hermes-4')).toBe(false)
    expect(modelPickerCommand('hermes-4', 'nous', false)).toBe('hermes-4 --provider nous --tui-session')
    expect(modelPickerCommand('hermes-4', 'nous', false, 'high')).toBe(
      'hermes-4 --provider nous --reasoning high --tui-session'
    )
  })
})
