import { describe, expect, it } from 'vitest'

import { buildCommandPaletteItems, filterCommandPaletteItems } from '../lib/commandPalette.js'
import type { SlashCatalog } from '../types.js'

const catalog: SlashCatalog = {
  canon: { '/m': '/model', '/models': '/model' },
  categories: [{ name: 'Session', pairs: [['/model', 'change model'], ['/resume', 'resume a prior session']] }],
  pairs: [['/model', 'change model'], ['/resume', 'resume a prior session']],
  skillCount: 0,
  sub: {}
}

describe('command palette helpers', () => {
  it('builds stable commands with help, usage, and aliases', () => {
    const items = buildCommandPaletteItems(
      [
        { aliases: ['m'], help: 'open model picker', name: 'model', usage: '/model [name]' },
        { help: 'toggle details', name: 'details' }
      ],
      catalog
    )

    expect(items.slice(0, 2)).toEqual([
      {
        aliases: ['m', 'models'],
        command: '/model',
        description: 'open model picker',
        lane: 'core',
        rune: '☤',
        searchText: '/model model open model picker /model [name] m models',
        usage: '/model [name]'
      },
      {
        aliases: [],
        command: '/details',
        description: 'toggle details',
        lane: 'core',
        rune: '☤',
        searchText: '/details details toggle details',
        usage: '/details'
      }
    ])
  })

  it('adds catalog-only commands without duplicating local ones', () => {
    const items = buildCommandPaletteItems([{ help: 'open model picker', name: 'model' }], catalog)

    expect(items.map(item => item.command)).toEqual(['/model', '/resume'])
    expect(items.find(item => item.command === '/resume')?.description).toBe('resume a prior session')
  })

  it('labels local and catalog commands with cockpit lanes', () => {
    const items = buildCommandPaletteItems([{ help: 'open model picker', name: 'model' }], catalog)

    expect(items.find(item => item.command === '/model')).toMatchObject({ lane: 'core', rune: '☤' })
    expect(items.find(item => item.command === '/resume')).toMatchObject({ lane: 'catalog', rune: '◇' })
  })

  it('keeps local commands ahead of catalog-only skill commands when filtered', () => {
    const items = buildCommandPaletteItems(
      [{ help: 'open the Aurora command palette', name: 'palette' }],
      { ...catalog, pairs: [...catalog.pairs, ['/parallel-cli', 'Optional vendor skill for Parallel CLI']] }
    )

    expect(filterCommandPaletteItems(items, 'pa').map(item => item.command)).toEqual(['/palette', '/parallel-cli'])
  })

  it('filters by command, description, usage, and alias text', () => {
    const items = buildCommandPaletteItems(
      [
        { aliases: ['m'], help: 'open model picker', name: 'model', usage: '/model [name]' },
        { help: 'browse skills hub', name: 'skills' },
        { help: 'resume a prior session', name: 'resume' }
      ],
      null
    )

    expect(filterCommandPaletteItems(items, 'pick').map(item => item.command)).toEqual(['/model'])
    expect(filterCommandPaletteItems(items, 'hub').map(item => item.command)).toEqual(['/skills'])
    expect(filterCommandPaletteItems(items, ' m ').map(item => item.command)).toEqual(['/model'])
  })
})
