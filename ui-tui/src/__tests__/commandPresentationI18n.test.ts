import { afterEach, expect, it, vi } from 'vitest'

import type { GatewayRpc } from '../app/interfaces.js'
import { coreCommands } from '../app/slash/commands/core.js'
import type { SlashRunCtx } from '../app/slash/types.js'
import { loadSlashCatalog } from '../app/slashCatalog.js'
import { resetUiState } from '../app/uiStore.js'
import { resetLocale } from '../i18n/runtime.js'
import type { SlashCatalog } from '../types.js'

import { activateZh } from './localeFixture.js'

afterEach(() => {
  resetLocale()
  resetUiState()
})

it('the session catalog refresh retains built-in presentation keys and custom descriptions', async () => {
  resetUiState()

  const response = {
    pairs: [
      ['/density', 'Toggle compact display mode'],
      ['/custom', 'My command']
    ],
    description_keys: { '/density': 'density' },
    categories: [{ name: 'Session', key: 'session', pairs: [['/density', 'Toggle compact display mode']] }]
  }

  const setCatalog = vi.fn()
  // SAFETY: loadSlashCatalog only calls commands.catalog, answered by its result fixture.
  await loadSlashCatalog(vi.fn(async () => response) as GatewayRpc, null, setCatalog)
  expect(setCatalog).toHaveBeenCalledWith(expect.objectContaining({ descriptionKeys: response.description_keys }))
})

it('help localizes the wire category key while preserving a custom command description', () => {
  activateZh()
  const panel = vi.fn()

  const catalog: SlashCatalog = {
    canon: {},
    sub: {},
    skillCount: 0,
    descriptionKeys: { '/density': 'density' },
    pairs: [],
    categories: [
      {
        name: 'Session',
        key: 'session',
        pairs: [
          ['/density', 'Toggle compact display mode'],
          ['/custom', 'My command']
        ]
      }
    ]
  }

  const ctx = {
    local: { catalog },
    transcript: { panel },
    ui: { theme: { brand: { helpHeader: 'help' } } }
  } as unknown as SlashRunCtx

  coreCommands.find(command => command.name === 'help')!.run('', ctx)
  expect(panel.mock.calls[0]).toEqual(
    expect.arrayContaining([
      expect.arrayContaining([
        expect.objectContaining({
          title: '会话',
          rows: [
            ['/density', '切换紧凑显示模式'],
            ['/custom', 'My command']
          ]
        })
      ])
    ])
  )
})
