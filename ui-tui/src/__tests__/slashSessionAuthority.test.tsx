import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

it('branch hydration replaces source authority only after a successful destination attachment', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-slash-branch-'))
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('HERMES_TUI_ACTIVE_SESSION_FILE', join(home, 'active'))
  resetUiState()

  const source = { model: 'test', skills: {}, tools: {}, stored_session_id: 'stored-source',
    execution_epoch: 'source-owner', execution_generation: 9, running: true }

  patchUiState({ sid: 'source', info: source, busy: true })
  const pending: Array<{ method: string; resolve: (value: any) => void }> = []

  const request = vi.fn((method: string) => method === 'session.close'
    ? Promise.resolve({}) : new Promise(resolve => pending.push({ method, resolve })))

  const setHistoryItems = vi.fn()
  const sys = vi.fn()
  let slash!: (command: string) => boolean

  function Harness() {
    const lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() },
      gw: { request }, rpc: request, scrollRef: { current: null }, panel: vi.fn(), sys,
      setHistoryItems, setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    slash = createSlashHandler({ slashFlightRef: { current: 0 }, gateway: { gw: { request }, rpc: request },
      local: {}, session: { ...lifecycle, setSessionStartedAt: vi.fn() }, transcript: { sys, setHistoryItems } } as any)

    return <Text>branch</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  const flush = () => new Promise(resolve => setImmediate(resolve))

  try {
    slash('/branch child')
    pending.shift()!.resolve({ session_id: 'branch', title: 'child' })
    await flush()
    expect(getUiState().sid).toBe('source')
    expect(request).not.toHaveBeenCalledWith('session.close', expect.anything())
    expect(pending[0]?.method).toBe('setup.status')
    pending.shift()!.resolve({ provider_configured: true })
    await flush()
    expect(pending[0]?.method).toBe('session.resume')
    pending.shift()!.resolve({ session_id: 'branch', stored_session_id: 'stored-branch', running: false,
      info: { ...source, stored_session_id: 'stored-branch', execution_epoch: 'branch-owner', execution_generation: 0, running: false },
      messages: [{ role: 'user', text: 'inherited history' }] })
    await flush()
    expect(getUiState()).toMatchObject({ sid: 'branch', busy: false, info: {
      stored_session_id: 'stored-branch', execution_epoch: 'branch-owner', execution_generation: 0 } })
    expect(setHistoryItems).toHaveBeenLastCalledWith(expect.arrayContaining([
      expect.objectContaining({ role: 'user', text: 'inherited history' })
    ]))
    expect(request).toHaveBeenCalledWith('session.close', { session_id: 'source' })

    // A branch response for a destination the user has left must not attach.
    slash('/branch stale')
    patchUiState({ sid: 'other' })
    pending.shift()!.resolve({ session_id: 'late-branch' })
    await flush()
    expect(pending).toHaveLength(0)
    expect(getUiState().sid).toBe('other')
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})
