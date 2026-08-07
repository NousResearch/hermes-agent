import { describe, expect, it } from 'vitest'

import { normalizeSlashSearchQuery, rankSlashItems, scoreSlashMenuItem, tokenizeSearchText } from './fuzzyScore.js'

describe('normalizeSlashSearchQuery', () => {
  it('trims, strips leading slashes, and lowercases', () => {
    expect(normalizeSlashSearchQuery(' /Model ')).toBe('model')
    expect(normalizeSlashSearchQuery('//help')).toBe('help')
    expect(normalizeSlashSearchQuery('plain')).toBe('plain')
  })
})

describe('tokenizeSearchText', () => {
  it('returns the full lowercased value plus alphanumeric word tokens', () => {
    expect(tokenizeSearchText('Commit & Push')).toEqual(['commit & push', 'commit', 'push'])
  })
})

describe('scoreSlashMenuItem', () => {
  const item = { aliases: ['recap', 'summary'], description: 'Turn session recaps on/off', id: 'recaps', label: 'recaps' }

  it('scores exact name matches at tier 0', () => {
    expect(scoreSlashMenuItem(item, 'recaps')).toBe(0)
  })

  it('scores exact alias matches at tier 0', () => {
    expect(scoreSlashMenuItem(item, 'summary')).toBe(0)
  })

  it('scores name prefixes at tier 1 and name substrings at tier 2', () => {
    expect(scoreSlashMenuItem(item, 'rec')).toBe(1)
    expect(scoreSlashMenuItem(item, 'caps')).toBe(2)
  })

  it('scores description matches at the +3 offset, below any name tier', () => {
    expect(scoreSlashMenuItem({ description: 'Turn session recaps on/off', id: 'other' }, 'session')).toBe(3)
    expect(scoreSlashMenuItem({ description: 'Turn session recaps on/off', id: 'other' }, 'sess')).toBe(4)
    expect(scoreSlashMenuItem({ description: 'Turn session recaps on/off', id: 'other' }, 'essio')).toBe(5)
  })

  it('prefers the name tier when both name and description match', () => {
    expect(scoreSlashMenuItem(item, 'recap')).toBe(0)
  })

  it('returns Infinity when nothing matches', () => {
    expect(scoreSlashMenuItem(item, 'zzz')).toBe(Number.POSITIVE_INFINITY)
  })
})

describe('rankSlashItems', () => {
  const apps = [
    { help: 'Show available commands', id: 'help' },
    { help: 'Start a countdown timer', id: 'clock' },
    { help: 'Select a model', id: 'models' }
  ]

  const toScoreItem = (app: (typeof apps)[number]) => ({ description: app.help, id: app.id })

  it('returns the list untouched for an empty query', () => {
    expect(rankSlashItems(apps, '/', toScoreItem)).toEqual(apps)
  })

  it('surfaces description matches the prefix filter would miss', () => {
    expect(rankSlashItems(apps, '/timer', toScoreItem).map(app => app.id)).toEqual(['clock'])
  })

  it('ranks name matches above description matches and drops non-matches', () => {
    const ranked = rankSlashItems(
      [{ help: 'model picker widget', id: 'gallery' }, ...apps],
      '/model',
      toScoreItem
    )

    expect(ranked.map(app => app.id)).toEqual(['models', 'gallery'])
  })

  it('keeps original order within a score tier', () => {
    const ranked = rankSlashItems(
      [
        { help: '', id: 'mod-b' },
        { help: '', id: 'mod-a' }
      ],
      '/mod',
      toScoreItem
    )

    expect(ranked.map(app => app.id)).toEqual(['mod-b', 'mod-a'])
  })
})
