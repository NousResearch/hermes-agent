import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

const flush = () => new Promise(resolve => setImmediate(resolve))

const COMMANDS = [
  '/mouse off',
  '/density on',
  '/details expanded',
  '/details tools hidden',
  '/focus on',
  '/statusbar off',
  '/battery on'
]

// Main persisted these through config.set; the shared owner serves the same keys profile-wide
// (gateway/session_tui_config.py), so a canonical TUI saves them too instead of dropping them.
function harness(isCanonical: boolean) {
  const request = vi.fn(async (_method: string, params: any) => ({ key: params?.key, value: params?.value }))

  const sys = vi.fn()

  const slash = createSlashHandler({
    composer: { enqueue: vi.fn() },
    gateway: { gw: { isCanonical, request }, rpc: request },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    slashFlightRef: { current: 0 },
    transcript: {
      page: vi.fn(),
      panel: vi.fn(),
      send: vi.fn(),
      setHistoryItems: vi.fn(),
      sys,
      trimLastExchange: (x: unknown[]) => x
    }
  } as any)

  return { notices: () => sys.mock.calls.map(([text]) => String(text)), request, slash }
}

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner' })
})

it("the shared gateway persists display preferences through config.set and prints main's text", async () => {
  const { notices, request, slash } = harness(true)

  for (const cmd of COMMANDS) {
    slash(cmd)
  }

  await flush()

  expect(request.mock.calls.filter(([method]) => method === 'config.set').map(([, params]) => params)).toEqual([
    { key: 'mouse', value: 'off' },
    { key: 'density', value: 'on' },
    { key: 'details_mode', value: 'expanded' },
    { key: 'details_mode.tools', value: 'hidden' },
    { key: 'focus', session_id: 'owner', value: 'on' },
    { key: 'statusbar', value: 'off' },
    { key: 'battery', value: 'on' }
  ])
  expect(getUiState()).toMatchObject({
    battery: true,
    compact: true,
    focusView: true,
    mouseTracking: 'off',
    statusBar: 'off'
  })
  expect(notices()).toEqual(expect.arrayContaining(['mouse tracking off', 'density on', 'status bar off']))
  expect(notices().some(text => text.includes('not saved'))).toBe(false)
})

it('a standalone backend still persists them through config.set', async () => {
  const { notices, request, slash } = harness(false)

  for (const cmd of COMMANDS) {
    slash(cmd)
  }

  await flush()

  expect(request.mock.calls.filter(([method]) => method === 'config.set').map(([, params]) => params.key)).toEqual([
    'mouse',
    'density',
    'details_mode',
    'details_mode.tools',
    'focus',
    'statusbar',
    'battery'
  ])
  expect(notices().some(text => text.includes('not saved'))).toBe(false)
})
