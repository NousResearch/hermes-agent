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

// The shared gateway's config.set accepts only busy/verbose/yolo/model, and these commands
// swallowed its refusal: the view changed, the notice said nothing, and the preference was gone
// on the next launch.
function harness(isCanonical: boolean) {
  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'config.set' && isCanonical) {
      throw new Error(`invalid_params: ${params.key}`)
    }

    return { value: params?.value }
  })

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

it('display preferences apply to the view and say they are not saved on the shared gateway', async () => {
  const { notices, request, slash } = harness(true)

  for (const cmd of COMMANDS) {
    slash(cmd)
  }

  await flush()

  expect(request.mock.calls.filter(([method]) => method === 'config.set')).toEqual([])
  expect(getUiState()).toMatchObject({
    battery: true,
    compact: true,
    focusView: true,
    mouseTracking: 'off',
    statusBar: 'off'
  })

  for (const name of ['mouse', 'density', 'details', 'focus', 'statusbar', 'battery']) {
    expect(notices()).toContain(`/${name} applies to this view only; not saved on the shared gateway`)
  }
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
