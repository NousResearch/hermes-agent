import { describe, expect, it } from 'vitest'

import { fuzzyRank, fuzzyScore, fuzzyScoreMulti } from './fuzzy.js'

describe('fuzzyScore', () => {
  it('matches a query as a subsequence (g4o → gpt-4o)', () => {
    expect(fuzzyScore('gpt-4o', 'g4o')).not.toBeNull()
    expect(fuzzyScore('gpt-4o', 'gpt')).not.toBeNull()
    expect(fuzzyScore('gpt-4o', '4o')).not.toBeNull()
  })

  it('returns null when characters are out of order or absent', () => {
    expect(fuzzyScore('gpt-4o', 'o4g')).toBeNull()
    expect(fuzzyScore('gpt-4o', 'xyz')).toBeNull()
    expect(fuzzyScore('gpt-4o', 'gptx')).toBeNull()
  })

  it('returns matched positions into the original target', () => {
    const m = fuzzyScore('gpt-4o', 'g4o')
    // g@0, 4@4, o@5
    expect(m?.positions).toEqual([0, 4, 5])
  })

  it('treats an empty query as a zero-score match', () => {
    expect(fuzzyScore('anything', '')).toEqual({ score: 0, positions: [] })
  })

  it('scores an exact match highest', () => {
    const exact = fuzzyScore('sonnet', 'sonnet')!.score
    const prefix = fuzzyScore('sonnet-extended', 'sonnet')!.score
    // s,o,n,n,e,t all present in order but scattered across word boundaries.
    const scattered = fuzzyScore('snorkel-online-nnet', 'sonnet')!.score

    expect(exact).toBeGreaterThan(prefix)
    expect(prefix).toBeGreaterThan(scattered)
  })

  it('ranks a prefix match above a scattered subsequence', () => {
    const prefix = fuzzyScore('gpt-4o-mini', 'gpt')!.score
    const scattered = fuzzyScore('a-g-p-t', 'gpt')!.score

    expect(prefix).toBeGreaterThan(scattered)
  })

  it('rewards word-boundary matches', () => {
    // `s4` matching the `s` of sonnet and the `4` after a dash
    const boundary = fuzzyScore('claude-sonnet-4', 'cs4')
    expect(boundary).not.toBeNull()
  })
})

describe('fuzzyScoreMulti', () => {
  it('requires every space-separated token to match (AND)', () => {
    expect(fuzzyScoreMulti('claude-sonnet-4', 'clad snnt')).not.toBeNull()
    expect(fuzzyScoreMulti('claude-sonnet-4', 'claude haiku')).toBeNull()
  })

  it('unions matched positions across tokens, sorted', () => {
    const m = fuzzyScoreMulti('claude-sonnet', 'son cla')
    expect(m).not.toBeNull()
    expect(m!.positions).toEqual([...m!.positions].sort((a, b) => a - b))
  })

  it('treats whitespace-only query as a zero-score match', () => {
    expect(fuzzyScoreMulti('x', '   ')).toEqual({ score: 0, positions: [] })
  })
})

describe('fuzzyRank', () => {
  const models = ['gpt-4o', 'gpt-4o-mini', 'claude-sonnet-4', 'claude-haiku', 'o1-preview']

  it('drops non-matching items and ranks matches by score', () => {
    const ranked = fuzzyRank(models, 'g4o', m => m)
    const ids = ranked.map(r => r.item)

    expect(ids).toContain('gpt-4o')
    expect(ids).toContain('gpt-4o-mini')
    expect(ids).not.toContain('claude-haiku')
    // Shorter exact-ish prefix should outrank the longer variant.
    expect(ids.indexOf('gpt-4o')).toBeLessThan(ids.indexOf('gpt-4o-mini'))
  })

  it('ranks son4 so a sonnet model surfaces', () => {
    const ranked = fuzzyRank(models, 'son4', m => m)
    expect(ranked[0]?.item).toBe('claude-sonnet-4')
  })

  it('returns all items in original order for an empty query', () => {
    const ranked = fuzzyRank(models, '', m => m)
    expect(ranked.map(r => r.item)).toEqual(models)
    expect(ranked.every(r => r.positions.length === 0)).toBe(true)
  })

  it('is stable for equal scores (original index tiebreak)', () => {
    const items = ['ab', 'ab', 'ab']

    const ranked = fuzzyRank(
      items.map((v, i) => ({ v, i })),
      'ab',
      x => x.v
    )

    expect(ranked.map(r => r.item.i)).toEqual([0, 1, 2])
  })

  it('matches across a derived key, not just the raw string', () => {
    const providers = [
      { slug: 'openai', name: 'OpenAI' },
      { slug: 'anthropic', name: 'Anthropic' }
    ]

    const ranked = fuzzyRank(providers, 'anth', p => `${p.name} ${p.slug}`)
    expect(ranked[0]?.item.slug).toBe('anthropic')
  })
})
