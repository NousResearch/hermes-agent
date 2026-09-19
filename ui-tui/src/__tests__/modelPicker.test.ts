import type { ModelOptionProvider } from '@hermes/shared/gateway-events'
import { describe, expect, it } from 'vitest'

import {
  filterSessionModelHopRows,
  providerIndexAfterClearingFilter,
  sessionModelHopRows,
  sessionModelHopSelection
} from '../components/modelPicker.js'

const provider = (slug: string, name = slug): ModelOptionProvider => ({ name, slug })

describe('ModelPicker provider filtering', () => {
  it('keeps the selected provider when clearing the provider filter', () => {
    const nous = provider('nous', 'Nous Portal')
    const ollama = provider('ollama-cloud', 'Ollama Cloud')

    const rows = [
      { name: nous.name, provider: nous },
      { name: ollama.name, provider: ollama }
    ]

    // With a provider-stage filter like "ollama", the selected row is index 0
    // in the filtered list, but index 1 in the full list after setFilter('').
    expect(providerIndexAfterClearingFilter(rows, ollama)).toBe(1)
  })

  it('returns -1 when provider is undefined', () => {
    const rows = [{ name: 'A', provider: provider('a') }]

    expect(providerIndexAfterClearingFilter(rows, undefined)).toBe(-1)
  })

  it('returns -1 when provider slug is not in rows', () => {
    const rows = [
      { name: 'A', provider: provider('a') },
      { name: 'B', provider: provider('b') }
    ]

    expect(providerIndexAfterClearingFilter(rows, provider('missing'))).toBe(-1)
  })

  it('returns -1 for empty rows', () => {
    expect(providerIndexAfterClearingFilter([], provider('a'))).toBe(-1)
  })

  it('finds the first match when multiple rows share a slug', () => {
    const p = provider('dup')

    const rows = [
      { name: 'First', provider: p },
      { name: 'Second', provider: p }
    ]

    expect(providerIndexAfterClearingFilter(rows, p)).toBe(0)
  })
})

describe('ModelPicker session-only model hop', () => {
  it('flattens authenticated provider models into one searchable catalog', () => {
    const rows = sessionModelHopRows([
      { authenticated: true, models: ['claude-sonnet', 'claude-opus'], name: 'Anthropic', slug: 'anthropic' },
      { authenticated: false, models: ['gpt-5'], name: 'OpenAI', slug: 'openai' },
      { models: ['kimi-k2'], name: 'Kimi', slug: 'kimi' }
    ])

    expect(rows).toEqual([
      { model: 'claude-sonnet', providerName: 'Anthropic', providerSlug: 'anthropic' },
      { model: 'claude-opus', providerName: 'Anthropic', providerSlug: 'anthropic' },
      { model: 'kimi-k2', providerName: 'Kimi', providerSlug: 'kimi' }
    ])
  })

  it('keeps the hop selection session-scoped, including its provider identity', () => {
    expect(sessionModelHopSelection({ model: 'claude-sonnet', providerName: 'Anthropic', providerSlug: 'anthropic' })).toBe(
      'claude-sonnet --provider anthropic --tui-session'
    )
  })

  it('filters the flat catalog by model alias or provider identity', () => {
    const rows = sessionModelHopRows([
      { models: ['k3'], name: 'Kimi Coding', slug: 'kimi-coding' },
      { models: ['gpt-5'], name: 'OpenAI', slug: 'openai' }
    ])

    expect(filterSessionModelHopRows(rows, 'kimi').map(row => row.model)).toEqual(['k3'])
    expect(filterSessionModelHopRows(rows, 'openai').map(row => row.model)).toEqual(['gpt-5'])
  })
})
