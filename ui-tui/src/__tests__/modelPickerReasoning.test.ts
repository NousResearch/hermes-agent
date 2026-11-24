import type { ModelOptionProvider } from '@hermes/shared/gateway-events'
import { describe, expect, it } from 'vitest'

import { draftModelNameFromArg } from '../components/activeSessionSwitcher.js'
import { modelPickerCommand, pickerOffersReasoning, reasoningPickerRows } from '../components/modelPicker.js'
import { applyLocale, messages, resetLocale } from '../i18n/runtime.js'

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
    expect(reasoningPickerRows().at(-1)?.value).toBe('')
    // The new-session draft label strips the effort flag like it strips --provider.
    expect(draftModelNameFromArg(modelPickerCommand('gpt-5.6', 'nous', false, 'low'))).toBe('gpt-5.6')
  })

  it('skips the step only when the catalog says the route has no reasoning control', () => {
    expect(pickerOffersReasoning(provider({ 'gpt-5.6': { fast: false, reasoning: false } }), 'gpt-5.6')).toBe(false)
    expect(pickerOffersReasoning(provider({ 'gpt-5.6': { fast: false, reasoning: true } }), 'gpt-5.6')).toBe(true)
    expect(pickerOffersReasoning(provider(undefined), 'gpt-5.6')).toBe(true)
    expect(pickerOffersReasoning(undefined, 'gpt-5.6')).toBe(true)
  })
  it('resolves the labelled effort rows against the active language at call time', () => {
    const en = messages().pickers.model.reasoning
    expect(reasoningPickerRows().at(-2)?.label).toBe(en.none)
    expect(reasoningPickerRows().at(-1)?.label).toBe(en.keepCurrent)

    applyLocale('pl', { lang: 'pl', surface: 'tui', messages: { 'pickers.model.reasoning.keepCurrent': 'Zachowaj' } })

    try {
      expect(reasoningPickerRows().at(-1)?.label).toBe('Zachowaj')
      // Ladder levels are `--reasoning` values, never translated.
      expect(reasoningPickerRows()[0]?.label).toBe(reasoningPickerRows()[0]?.value)
    } finally {
      resetLocale()
    }

    expect(reasoningPickerRows().at(-1)?.label).toBe(en.keepCurrent)
  })
})
