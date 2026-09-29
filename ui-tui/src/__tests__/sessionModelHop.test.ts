import type { ModelOptionProvider, ModelOptionsResult } from '@hermes/shared/gateway-events'
import { beforeEach, describe, expect, it } from 'vitest'

import {
  buildSessionModelHopRows,
  filterSessionModelHopRows,
  sessionModelHopCurrentIndex
} from '../components/sessionModelHop.js'
import {
  cachedModelOptions,
  invalidateModelOptions,
  rememberModelOptions,
  resetModelOptionsCacheForTests
} from '../lib/modelOptionsCache.js'

const provider = (
  slug: string,
  models: string[],
  authenticated: boolean | undefined = true,
  isCurrent = false
): ModelOptionProvider => ({ authenticated, is_current: isCurrent, models, name: slug.toUpperCase(), slug })

describe('SessionModelHop catalog', () => {
  const rows = buildSessionModelHopRows([
    provider('nous', ['hermes-4', 'claude-sonnet-4.6']),
    provider('openrouter', ['anthropic/claude-sonnet-4.6']),
    provider('hidden', ['x'], false)
  ])

  it('flattens configured providers into unambiguous provider/model selectors', () => {
    expect(rows.map(row => row.selector)).toEqual([
      'nous/hermes-4',
      'nous/claude-sonnet-4.6',
      'openrouter/anthropic/claude-sonnet-4.6'
    ])
  })

  it('filters locally across provider and model fragments', () => {
    expect(filterSessionModelHopRows(rows, 'nous hermes').map(row => row.selector)).toEqual(['nous/hermes-4'])
    expect(filterSessionModelHopRows(rows, 'openrouter claude').map(row => row.selector)).toEqual([
      'openrouter/anthropic/claude-sonnet-4.6'
    ])
  })

  it('supports subsequence fuzzy shorthand while typing', () => {
    expect(filterSessionModelHopRows(rows, 'son4').map(row => row.selector)).toContain(
      'nous/claude-sonnet-4.6'
    )
    expect(filterSessionModelHopRows(rows, 'hrms').map(row => row.selector)).toEqual(['nous/hermes-4'])
  })

  it('finds the current row by full selector or model id', () => {
    expect(sessionModelHopCurrentIndex(rows, 'nous/hermes-4')).toBe(0)
    expect(sessionModelHopCurrentIndex(rows, 'anthropic/claude-sonnet-4.6')).toBe(2)
  })

  it('returns -1 when the current model is absent instead of aliasing the first result', () => {
    expect(sessionModelHopCurrentIndex(rows, 'missing-model')).toBe(-1)
  })

  it('uses the backend current-provider hint when model ids collide', () => {
    const duplicateRows = buildSessionModelHopRows([
      provider('nous', ['shared-model']),
      provider('openrouter', ['shared-model'], true, true)
    ])

    expect(sessionModelHopCurrentIndex(duplicateRows, 'shared-model')).toBe(1)
  })
})

describe('model options receipt cache', () => {
  const result = { model: 'hermes-4', providers: [] } as ModelOptionsResult

  beforeEach(() => resetModelOptionsCacheForTests())

  it('reuses a fresh per-session receipt and expires it', () => {
    rememberModelOptions('s1', result, 100)
    expect(cachedModelOptions('s1', 10_000)).toBe(result)
    expect(cachedModelOptions('s1', 10_101)).toBeNull()
  })

  it('does not leak catalogs across sessions and supports explicit invalidation', () => {
    rememberModelOptions('s1', result, 100)
    expect(cachedModelOptions('s2', 101)).toBeNull()
    invalidateModelOptions('s1')
    expect(cachedModelOptions('s1', 101)).toBeNull()
  })
})
