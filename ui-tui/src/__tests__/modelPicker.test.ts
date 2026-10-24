import type { ModelOptionProvider } from '@hermes/shared/gateway-events'
import { describe, expect, it } from 'vitest'

import { providerIndexAfterClearingFilter, providerStageRows } from '../components/modelPicker.js'

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

  it('finds the first match when multiple rows share a slug', () => {
    const p = provider('dup')

    const rows = [
      { name: 'First', provider: p },
      { name: 'Second', provider: p }
    ]

    expect(providerIndexAfterClearingFilter(rows, p)).toBe(0)
  })
})

describe('ModelPicker step-1 search', () => {
  it('ranks step-1 rows so Enter lands where the user means', () => {
    const anthropic = { ...provider('anthropic', 'Anthropic'), authenticated: false, models: ['claude-opus-5'] }
    const openrouter = { ...provider('openrouter', 'OpenRouter'), models: ['anthropic/claude-opus-5', 'openai/gpt-6'] }
    const zen = { ...provider('opencode-zen', 'OpenCode Zen'), models: ['zen-1'] }
    const copilot = { ...provider('copilot', 'GitHub Copilot'), is_current: true, models: ['openai/gpt-6'] }
    const rows = [openrouter, anthropic, zen, copilot].map(p => ({ name: p.name, provider: p }))
    const modelRows = (query: string) => providerStageRows(rows, query).filter(row => row.kind === 'model')

    // A query naming a provider stays on it, even when its own or an aggregator's model ids match better.
    expect(providerStageRows(rows, 'anthropic')[0]).toEqual({ kind: 'provider', ...rows[1] })
    expect(providerStageRows(rows, 'zen')[0]).toEqual({ kind: 'provider', ...rows[2] })
    // Models only from configured providers; an id several providers serve prefers the current one.
    expect(modelRows('claude-opus-5').map(row => row.provider.slug)).toEqual(['openrouter'])
    expect(modelRows('openai/gpt-6').map(row => row.provider.slug)).toEqual(['copilot', 'openrouter'])
    // No model's id contains `gpt-6-pro`; it must not reach `openai/gpt-6` via letters in "OpenRouter".
    expect(modelRows('gpt-6-pro')).toEqual([])
  })
})
