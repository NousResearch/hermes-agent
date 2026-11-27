import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

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

it('/save renders through session.export and Ink writes the file itself (T10)', async () => {
  const dir = mkdtempSync(join(process.env.TMPDIR || tmpdir(), 'ink-save-'))
  const saved = process.env.HERMES_HOME

  process.env.HERMES_HOME = dir

  try {
    const exported = { content: '{"messages": []}', filename: 'hermes_owner.json', format: 'json' }
    const { request, slash, sys } = harness({ 'session.export': exported })

    slash('/save')
    await flush()
    await flush()

    expect(request).toHaveBeenCalledWith('session.export', { format: 'json', redact: false, session_id: 'owner' })
    expect(request.mock.calls.some(([method]) => method === 'session.save')).toBe(false)
    const [file] = readdirSync(join(dir, 'sessions', 'saved'))
    expect(file).toBe('hermes_owner.json') // the owner's name, so Ink and Desktop agree
    expect(readFileSync(join(dir, 'sessions', 'saved', file!), 'utf8')).toBe('{"messages": []}')
    expect(notices(sys)).toContain(`conversation saved to: ${join(dir, 'sessions', 'saved', file!)}`)

    const md = { content: '# t', filename: 'hermes_owner.md', format: 'md' }
    const second = harness({ 'session.export': md })

    second.slash(`/save md ${join(dir, 'out.md')} redact`)
    await flush()
    await flush()
    expect(second.request).toHaveBeenCalledWith('session.export', { format: 'md', redact: true, session_id: 'owner' })
    expect(readFileSync(join(dir, 'out.md'), 'utf8')).toBe('# t')
  } finally {
    process.env.HERMES_HOME = saved
    rmSync(dir, { force: true, recursive: true })
  }
})

it("/save resolves a relative name in the user's folder, not the install cwd Node runs in", async () => {
  const dir = mkdtempSync(join(process.env.TMPDIR || tmpdir(), 'ink-save-cwd-'))

  try {
    patchUiState({ info: { cwd: dir } as any })
    const { slash } = harness({ 'session.export': { content: 'REL', filename: 'x.md', format: 'md' } })

    slash('/save md rel-out.md')
    await flush()
    await flush()
    expect(readFileSync(join(dir, 'rel-out.md'), 'utf8')).toBe('REL')
  } finally {
    rmSync(dir, { force: true, recursive: true })
  }
})

it('/save still writes the file when another slash command starts before the export returns', async () => {
  const dir = mkdtempSync(join(process.env.TMPDIR || tmpdir(), 'ink-save-race-'))
  let finish!: (value: unknown) => void

  try {
    const { request, slash } = harness({ 'shared_metrics.slash_command': {} })

    request.mockImplementation(async (method: string) =>
      method === 'session.export' ? new Promise(resolve => (finish = resolve)) : {}
    )
    slash(`/save md ${join(dir, 'race.md')}`)
    slash('/timestamps status')
    finish({ content: 'RACE', filename: 'x.md', format: 'md' })
    await flush()
    await flush()
    expect(readFileSync(join(dir, 'race.md'), 'utf8')).toBe('RACE')
  } finally {
    rmSync(dir, { force: true, recursive: true })
  }
})

it('/save still reports a failed write after another slash command started', async () => {
  const dir = mkdtempSync(join(process.env.TMPDIR || tmpdir(), 'ink-save-fail-'))
  let finish!: (value: unknown) => void

  try {
    writeFileSync(join(dir, 'file'), '')
    const { request, slash, sys } = harness({ 'shared_metrics.slash_command': {} })

    request.mockImplementation(async (method: string) =>
      method === 'session.export' ? new Promise(resolve => (finish = resolve)) : {}
    )
    slash(`/save md ${join(dir, 'file', 'out.md')}`)
    slash('/timestamps status')
    finish({ content: 'X', filename: 'x.md', format: 'md' })
    await flush()
    await flush()
    expect(notices(sys).some(n => n.startsWith('failed to save: '))).toBe(true)
  } finally {
    rmSync(dir, { force: true, recursive: true })
  }
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

  for (const cmd of [
    '/rollback',
    '/rollback diff 1',
    '/rollback 1 src/a.py',
    '/rollback 1',
    '/rollback 1 --all',
    '/rollback 1 src/a.py --force'
  ]) {
    slash(cmd)
    await flush()
  }

  expect(request.mock.calls.filter(([method]) => String(method).startsWith('rollback.'))).toEqual([
    ['rollback.list', { session_id: 'owner' }],
    ['rollback.diff', { hash: '1', session_id: 'owner' }],
    ['rollback.restore', { file_path: 'src/a.py', hash: '1', session_id: 'owner' }],
    ['rollback.restore', { hash: '1', session_id: 'owner' }],
    // Main's --all/--force reverts hand-edits too; it is a flag, never part of the file path.
    ['rollback.restore', { all: true, hash: '1', session_id: 'owner' }],
    ['rollback.restore', { file_path: 'src/a.py', hash: '1', session_id: 'owner' }]
  ])
  expect(panel).toHaveBeenCalledWith('Rollback checkpoints', [{ rows: [['1. abcdef1234', 't1 · before write_file']] }])
  expect(page).toHaveBeenCalledWith(' a.txt | 2 +-\n\n-old\n+new', expect.any(String))
  expect(notices(sys)).toEqual([
    'rollback restored src/a.py: before write_file',
    'rollback restored workspace: before write_file',
    'rollback restored workspace: before write_file',
    'rollback restored src/a.py: before write_file'
  ])
  // A whole-tree restore rewound the transcript (history_removed > 0): the view drops the exchange.
  expect(setHistoryItems).toHaveBeenCalled()
})

it("/rollback restore names the hand-edits it kept, oversize skips and failed deletions, as main's CLI does", async () => {
  const { slash, sys } = harness({
    'rollback.restore': {
      failed_deletes: ['locked.txt'],
      history_removed: 0,
      reason: 'before write_file',
      skipped_oversize: ['big.bin'],
      skipped_user_edits: ['a.md', 'b.md', 'c.md', 'd.md', 'e.md', 'f.md', 'g.md'],
      success: true
    }
  })

  slash('/rollback 1')
  await flush()

  // The gateway-backed classic CLI's lines (hermes_cli/gateway_rollback.py), with its --all hint.
  expect(notices(sys)).toEqual([
    'rollback restored workspace: before write_file',
    '↷ Kept your hand-edits: a.md, b.md, c.md, d.md, e.md (+2 more)\nUse /rollback <N> --all to restore those too.',
    '↷ Kept (too large for checkpoints, no stored copy to revert to): big.bin',
    '⚠️ Could not remove (left in place): locked.txt'
  ])
})
