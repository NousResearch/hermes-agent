import { describe, expect, it } from 'vitest'

import { modelOptionsRequestParams, providerIndexAfterClearingFilter } from '../components/modelPicker.js'
import type { ModelOptionProvider } from '../gatewayTypes.js'

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

describe('ModelPicker model.options params', () => {
  it('requests the full provider universe by default', () => {
    expect(modelOptionsRequestParams('sess-1', false, false)).toEqual({
      session_id: 'sess-1',
      include_unconfigured: true
    })
  })

  it('requests explicit configured providers when hiding unconfigured providers', () => {
    expect(modelOptionsRequestParams('sess-1', false, true)).toEqual({
      session_id: 'sess-1',
      explicit_only: true
    })
  })

  it('preserves refresh while requesting all providers', () => {
    expect(modelOptionsRequestParams(null, true, false)).toEqual({
      refresh: true,
      include_unconfigured: true
    })
  })

  it('preserves refresh while requesting configured providers', () => {
    expect(modelOptionsRequestParams(null, true, true)).toEqual({
      refresh: true,
      explicit_only: true
    })
  })
})
