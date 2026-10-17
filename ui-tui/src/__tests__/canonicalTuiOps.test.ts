import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// The shared gateway serves Ink's /save, /reload, /agents pause|resume and /rollback through the
// owner verbs in gateway/session_tui_ops.py: each command reaches its verb and renders main's text.

const flush = () => new Promise(resolve => setImmediate(resolve))

function harness(results: Record<string, unknown> = {}) {
  const messages = [
    { role: 'user', text: 'hello' },
    { role: 'assistant', text: 'hi there' }
  ]

  const request = vi.fn(async (method: string, _params?: unknown) => {
    if (method in results) {
      return results[method]
    }

    throw new Error(`unknown_method: ${method}`)
  })

  const sys = vi.fn()
  const page = vi.fn()
  const panel = vi.fn()
  const setHistoryItems = vi.fn()

  const ctx = {
    composer: { enqueue: vi.fn() },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    local: { getHistoryItems: () => messages, getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    slashFlightRef: { current: 0 },
    transcript: { page, panel, send: vi.fn(), setHistoryItems, sys, trimLastExchange: (x: unknown[]) => x }
  }

  return { page, panel, request, setHistoryItems, slash: createSlashHandler(ctx as any), sys }
}

const notices = (sys: ReturnType<typeof vi.fn>) => sys.mock.calls.map(([text]) => String(text))

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner' })
})

it('/save calls session.save on the shared gateway and reports the saved file', async () => {
  const { request, slash, sys } = harness({ 'session.save': { file: '/home/u/.hermes/sessions/saved/x.json' } })

  slash('/save')
  await flush()

  expect(request).toHaveBeenCalledWith('session.save', { session_id: 'owner' })
  expect(notices(sys)).toContain('conversation saved to: /home/u/.hermes/sessions/saved/x.json')
})

it('/reload calls reload.env for this session on the shared gateway and reports the count', async () => {
  const { request, slash, sys } = harness({ 'reload.env': { updated: 2 } })

  slash('/reload')
  await flush()

  expect(request).toHaveBeenCalledWith('reload.env', { session_id: 'owner' })
  expect(notices(sys)).toContain('reloaded .env (2 vars updated)')
})

it("/agents pause and resume call delegation.pause on the shared gateway with main's notices", async () => {
  const { request, slash, sys } = harness({ 'delegation.pause': { paused: true } })

  slash('/agents pause')
  await flush()

  expect(request).toHaveBeenCalledWith('delegation.pause', { paused: true })

  request.mockImplementation(async () => ({ paused: false }))
  slash('/agents resume')
  await flush()

  expect(request).toHaveBeenCalledWith('delegation.pause', { paused: false })
  expect(notices(sys)).toEqual(['delegation · paused', 'delegation · resumed'])
})

it("/rollback list, diff and restore call the owner's rollback verbs with main's rendering", async () => {
  const { page, panel, request, setHistoryItems, slash, sys } = harness({
    'rollback.list': {
      enabled: true,
      checkpoints: [{ hash: 'abcdef1234567', timestamp: 't1', message: 'before write_file' }]
    },
    'rollback.diff': { stat: ' a.txt | 2 +-', diff: '-old\n+new' },
    'rollback.restore': { success: true, reason: 'before write_file', history_removed: 2 }
  })

  for (const cmd of ['/rollback', '/rollback diff 1', '/rollback 1 src/a.py', '/rollback 1']) {
    slash(cmd)
    await flush()
  }

  expect(request.mock.calls.filter(([method]) => String(method).startsWith('rollback.'))).toEqual([
    ['rollback.list', { session_id: 'owner' }],
    ['rollback.diff', { hash: '1', session_id: 'owner' }],
    ['rollback.restore', { file_path: 'src/a.py', hash: '1', session_id: 'owner' }],
    ['rollback.restore', { hash: '1', session_id: 'owner' }]
  ])
  expect(panel).toHaveBeenCalledWith('Rollback checkpoints', [{ rows: [['1. abcdef1234', 't1 · before write_file']] }])
  expect(page).toHaveBeenCalledWith(' a.txt | 2 +-\n\n-old\n+new', expect.any(String))
  expect(notices(sys)).toEqual([
    'rollback restored src/a.py: before write_file',
    'rollback restored workspace: before write_file'
  ])
  // A whole-tree restore rewound the transcript (history_removed > 0): the view drops the exchange.
  expect(setHistoryItems).toHaveBeenCalled()
})
