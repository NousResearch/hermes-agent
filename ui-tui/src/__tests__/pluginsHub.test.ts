import { describe, expect, it } from 'vitest'

import { classifyPluginsListKey, rankPluginRows } from '../components/pluginsHub.js'

describe('plugins hub fuzzy filtering', () => {
  const rows = [
    { description: 'Sync issues and pull requests', name: 'github', source: 'user', status: 'enabled' },
    { description: 'Manage team messages', name: 'slack-tools', source: 'bundled', status: 'disabled' },
    { description: 'Inspect calendar events', name: 'agenda', source: 'user', status: 'enabled' }
  ]

  it('keeps source order for an empty query', () => {
    expect(rankPluginRows(rows, '')).toEqual(rows)
  })

  it('matches and ranks across plugin metadata', () => {
    expect(rankPluginRows(rows, 'slk').map(row => row.name)).toEqual(['slack-tools'])
    expect(rankPluginRows(rows, 'pull request').map(row => row.name)).toEqual(['github'])
    expect(rankPluginRows(rows, 'bundled')[0]?.name).toBe('slack-tools')
  })
})

describe('plugins hub list key gating', () => {
  it('preserves q, Space, and number shortcuts before filtering', () => {
    expect(classifyPluginsListKey('q', {}, false)).toEqual({ kind: 'close' })
    expect(classifyPluginsListKey(' ', {}, false)).toEqual({ kind: 'select' })
    expect(classifyPluginsListKey('1', {}, false)).toEqual({ kind: 'quick', n: 1 })
  })

  it('routes the same printable keys into an active filter', () => {
    expect(classifyPluginsListKey('q', {}, true)).toEqual({ ch: 'q', kind: 'append' })
    expect(classifyPluginsListKey(' ', {}, true)).toEqual({ ch: ' ', kind: 'append' })
    expect(classifyPluginsListKey('1', {}, true)).toEqual({ ch: '1', kind: 'append' })
  })

  it('clears before closing and keeps navigation available', () => {
    expect(classifyPluginsListKey('', { escape: true }, true)).toEqual({ kind: 'clearFilter' })
    expect(classifyPluginsListKey('', { escape: true }, false)).toEqual({ kind: 'close' })
    expect(classifyPluginsListKey('', { downArrow: true }, true)).toEqual({ kind: 'down' })
    expect(classifyPluginsListKey('', { tab: true }, true)).toEqual({ kind: 'toggleScope' })
  })

  it('edits an active filter without capturing unrelated chords', () => {
    expect(classifyPluginsListKey('', { backspace: true }, true)).toEqual({ kind: 'backspace' })
    expect(classifyPluginsListKey('u', { ctrl: true }, true)).toEqual({ kind: 'clearFilter' })
    expect(classifyPluginsListKey(String.fromCharCode(21), { ctrl: true }, true)).toEqual({ kind: 'clearFilter' })
    expect(classifyPluginsListKey('g', { ctrl: true }, true)).toEqual({ kind: 'ignore' })
  })
})
