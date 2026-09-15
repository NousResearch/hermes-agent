import type { ModelOptionProvider } from '@hermes/shared/gateway-events'
import { describe, expect, it } from 'vitest'

import {
  buildModelHopRows,
  filterModelHopRows,
  hopCurrentIndex,
  hopIsCurrent,
  paintHits,
  providerIndexAfterClearingFilter,
  searchAppend
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

describe('ModelPicker hop catalog', () => {
  const nous = provider('nous', 'Nous Portal')
  const openrouter = provider('openrouter', 'OpenRouter')
  nous.models = ['claude-sonnet-4.6', 'hermes-4']
  openrouter.models = ['anthropic/claude-sonnet-4.6']

  it('lists every model as provider/id', () => {
    const rows = buildModelHopRows([nous, openrouter], ['Nous Portal', 'OpenRouter'])
    expect(rows.map(row => row.selector)).toEqual([
      'nous/claude-sonnet-4.6',
      'nous/hermes-4',
      'openrouter/anthropic/claude-sonnet-4.6'
    ])
  })

  it('filters like omp /switch: provider then / then model', () => {
    const rows = buildModelHopRows([nous, openrouter], ['Nous Portal', 'OpenRouter'])
    expect(filterModelHopRows(rows, 'nous/').map(row => row.selector)).toEqual([
      'nous/claude-sonnet-4.6',
      'nous/hermes-4'
    ])
    expect(filterModelHopRows(rows, 'nous/hermes').map(row => row.model)).toEqual(['hermes-4'])
    expect(filterModelHopRows(rows, 'openrouter/').map(row => row.selector)).toEqual([
      'openrouter/anthropic/claude-sonnet-4.6'
    ])
  })

  it('fuzzy-matches model fragments without a provider prefix', () => {
    const rows = buildModelHopRows([nous, openrouter], ['Nous Portal', 'OpenRouter'])
    expect(filterModelHopRows(rows, 'son4').map(row => row.model)).toEqual([
      'anthropic/claude-sonnet-4.6',
      'claude-sonnet-4.6'
    ])
    expect(filterModelHopRows(rows, 'hrms').map(row => row.selector)).toEqual(['nous/hermes-4'])
  })

  it('AND-matches provider/model tokens across nested ids', () => {
    const rows = buildModelHopRows([nous, openrouter], ['Nous Portal', 'OpenRouter'])
    expect(filterModelHopRows(rows, 'openrouter/claude').map(row => row.selector)).toEqual([
      'openrouter/anthropic/claude-sonnet-4.6'
    ])
  })
})

describe('hop current + paste', () => {
  it('stars by selector or current-provider id', () => {
    const nous = provider('nous')
    nous.is_current = true
    nous.models = ['hermes-4']
    const or = provider('openrouter')
    or.models = ['hermes-4']
    const rows = buildModelHopRows([nous, or], ['n', 'o'])
    expect(rows.filter(r => hopIsCurrent(r, 'nous/hermes-4')).map(r => r.selector)).toEqual(['nous/hermes-4'])
    expect(rows.filter(r => hopIsCurrent(r, 'hermes-4')).map(r => r.selector)).toEqual(['nous/hermes-4'])
    expect(hopCurrentIndex(rows, 'nous/hermes-4')).toBe(0)
  })

  it('appends paste, ignores controls', () => {
    expect(searchAppend('', 'nous/hermes-4')).toBe('nous/hermes-4')
    expect(searchAppend('n', '\t')).toBe('n')
  })
})

describe('paintHits', () => {
  it('marks typed subsequence chars', () => {
    expect(paintHits('claude-sonnet-4.6', 'son4').filter(p => p.hit).map(p => p.t)).toEqual(['son', '4'])
  })

  it('is a no-op for an empty query', () => {
    expect(paintHits('nous/hermes-4', '')).toEqual([{ t: 'nous/hermes-4', hit: false }])
  })
})

