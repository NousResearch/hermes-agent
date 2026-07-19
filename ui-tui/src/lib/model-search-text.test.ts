import { describe, expect, it } from 'vitest'

import { fuzzyRank } from './fuzzy.js'
import { modelSearchText } from './model-search-text.js'

describe('modelSearchText', () => {
  it('keeps ordinary model ids unchanged', () => {
    expect(modelSearchText('kimi-k2.6')).toBe('kimi-k2.6')
    expect(modelSearchText('glm-5.2')).toBe('glm-5.2')
  })

  it('adds kimi aliases for the bare Kimi Coding k3 wire id', () => {
    expect(modelSearchText('k3')).toBe('k3 kimi-k3 kimi')
    expect(modelSearchText('K3')).toBe('K3 kimi-k3 kimi')
  })
})

describe('model picker search with aliases', () => {
  const models = [
    'kimi-k2.6',
    'kimi-k2.5',
    'k3',
    'kimi-for-coding',
  ]

  it('surfaces k3 when the user searches kimi', () => {
    const ranked = fuzzyRank(models, 'kimi', modelSearchText).map(r => r.item)
    expect(ranked).toContain('k3')
  })

  it('still finds k3 by its wire id', () => {
    const ranked = fuzzyRank(models, 'k3', modelSearchText).map(r => r.item)
    expect(ranked).toEqual(['k3'])
  })

  it('does not invent k3 for unrelated queries', () => {
    const ranked = fuzzyRank(models, 'glm', modelSearchText).map(r => r.item)
    expect(ranked).toEqual([])
  })
})
