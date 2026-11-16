import { PassThrough } from 'node:stream'

import { renderSync, useStdin } from '@hermes/ink'
import React from 'react'
import { beforeEach, expect, it, vi } from 'vitest'

import { handleDashboardStop, useDashboardStop } from '../app/dashboardStop.js'
import { $inputSelection } from '../app/inputSelectionStore.js'
import { $overlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

beforeEach(() => {
  resetUiState()
  turnController.interrupted = false
  patchUiState({ sid: 'session-a', busy: true, info: { model: 'm', skills: {}, tools: {}, execution_epoch: 'epoch-a', execution_generation: 4 } })
})

it('routes the bound dashboard stop to the real generation-fenced turn controller', async () => {
  const request = vi.fn(async () => ({ execution_state: 'running', execution_generation: 4 }))
  const sys = vi.fn()
  await handleDashboardStop({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', sid: 'session-a', epoch: 'epoch-a', generation: 4, nonce: 'a'.repeat(32) }) },
    { gw: { isCanonical: true, request } as unknown as Parameters<typeof handleDashboardStop>[1]['gw'], appendMessage: vi.fn(), sys })
  expect(request).toHaveBeenCalledWith('session.interrupt', { session_id: 'session-a', execution_generation: 4 })
  expect(turnController.interrupted).toBe(true)
})

it('rejects a stop with an invalid query nonce', async () => {
  const request = vi.fn(async () => ({ execution_state: 'running', execution_generation: 4 }))
  await handleDashboardStop({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', sid: 'session-a', epoch: 'epoch-a', generation: 4, nonce: 'invalid' }) },
    { gw: { isCanonical: true, request } as unknown as Parameters<typeof handleDashboardStop>[1]['gw'], appendMessage: vi.fn(), sys: vi.fn() })
  expect(request).not.toHaveBeenCalled()
})

it.each([
  { sid: 'other', epoch: 'epoch-a', generation: 4 },
  { sid: 'session-a', epoch: 'other', generation: 4 },
  { sid: 'session-a', epoch: 'epoch-a', generation: 5 }
])('ignores stale binding $sid $epoch $generation', async binding => {
  const request = vi.fn()
  await handleDashboardStop({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', ...binding, nonce: 'a'.repeat(32) }) },
    { gw: { isCanonical: true, request }, appendMessage: vi.fn(), sys: vi.fn() })
  expect(request).not.toHaveBeenCalled()
})

it.each(['idle', 'interrupted', 'noncanonical'])('ignores stop when %s', async state => {
  if (state === 'idle') {patchUiState({ busy: false })}

  if (state === 'interrupted') {turnController.interrupted = true}
  const request = vi.fn()
  await handleDashboardStop({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', sid: 'session-a', epoch: 'epoch-a', generation: 4, nonce: 'a'.repeat(32) }) },
    { gw: { isCanonical: state !== 'noncanonical', request }, appendMessage: vi.fn(), sys: vi.fn() })
  expect(request).not.toHaveBeenCalled()
})

it('does not interrupt a successor when an acknowledgement arrives late', async () => {
  let resolve!: (value: { execution_state: string; execution_generation: number }) => void
  const request = vi.fn(() => new Promise<{ execution_state: string; execution_generation: number }>(r => { resolve = r }))

  const pending = handleDashboardStop({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', sid: 'session-a', epoch: 'epoch-a', generation: 4, nonce: 'a'.repeat(32) }) },
    { gw: { isCanonical: true, request } as unknown as Parameters<typeof handleDashboardStop>[1]['gw'], appendMessage: vi.fn(), sys: vi.fn() })

  patchUiState({ info: { model: 'm', skills: {}, tools: {}, execution_epoch: 'epoch-a', execution_generation: 5 } })
  resolve({ execution_state: 'running', execution_generation: 4 })
  await pending
  expect(turnController.interrupted).toBe(false)
})


it('mounted adapter replies to queries and republishes after a failed stop', async () => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  Object.assign(stdout, { columns: 80, rows: 20, isTTY: false })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  let output = ''
  stdout.on('data', chunk => { output += String(chunk) })
  let emit: ((response: { type: string; code: number; data: string }) => void) | undefined
  const request = vi.fn(async () => { throw new Error('rpc failed') })
  const gw = { isCanonical: true, request }
  const appendMessage = vi.fn()
  const sys = vi.fn()

  function Harness() {
    const { inputEmitter } = useStdin()

    emit = response => { inputEmitter.emit('terminalresponse', response) }
    useDashboardStop({ enabled: true, stdout: stdout as unknown as NodeJS.WriteStream, gw, appendMessage, sys })

    return null
  }

  const instance = renderSync(React.createElement(Harness), { stdin, stdout, stderr, exitOnCtrlC: false })
  await vi.waitFor(() => expect(emit).toBeTypeOf('function'), { timeout: 2000 })
  const nonce = 'b'.repeat(32)
  emit?.({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'query', nonce }) })
  expect(output).toContain(JSON.stringify({ action: 'state', bound: { sid: 'session-a', epoch: 'epoch-a', generation: 4 }, nonce }))
  output = ''
  emit?.({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'query', nonce: 'invalid' }) })
  expect(output).toBe('')
  patchUiState({ busy: false })
  emit?.({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'query', nonce }) })
  expect(output).toContain(JSON.stringify({ action: 'state', bound: null, nonce }))
  patchUiState({ busy: true })
  output = ''
  emit?.({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', sid: 'session-a', epoch: 'epoch-a', generation: 4, nonce }) })
  await vi.waitFor(() => expect(output).toContain(JSON.stringify({ action: 'state', bound: { sid: 'session-a', epoch: 'epoch-a', generation: 4 }, nonce })), { timeout: 2000 })
  expect(request).toHaveBeenCalledOnce()
  expect(output).toContain(JSON.stringify({ action: 'state', bound: { sid: 'session-a', epoch: 'epoch-a', generation: 4 }, nonce }))
  instance.unmount()
})


it.each([
  { type: 'osc', code: 6973, data: '{' },
  { type: 'osc', code: 6972, data: '{}' },
  { type: 'csi', code: 6973, data: '{}' },
  { type: 'osc', code: 6973, data: '[]' }
])('ignores invalid terminal frame', async frame => {
  const request = vi.fn()
  await handleDashboardStop(frame, { gw: { isCanonical: true, request }, appendMessage: vi.fn(), sys: vi.fn() })
  expect(request).not.toHaveBeenCalled()
})

it('does not interrupt after navigation while the RPC is pending', async () => {
  let resolve!: (value: { execution_state: string; execution_generation: number }) => void
  const request = vi.fn(() => new Promise<{ execution_state: string; execution_generation: number }>(r => { resolve = r }))

  const pending = handleDashboardStop({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', sid: 'session-a', epoch: 'epoch-a', generation: 4, nonce: 'a'.repeat(32) }) },
    { gw: { isCanonical: true, request } as unknown as Parameters<typeof handleDashboardStop>[1]['gw'], appendMessage: vi.fn(), sys: vi.fn() })

  patchUiState({ sid: 'session-b' })
  resolve({ execution_state: 'running', execution_generation: 4 })
  await pending
  expect(turnController.interrupted).toBe(false)
})


it('preserves the composer selection and overlay on a successful dashboard stop', async () => {
  const selection = { value: 'unfinished draft', start: 2, end: 8, cursor: 8, clear: vi.fn(), collapseToEnd: vi.fn(), copy: vi.fn(), cut: vi.fn() }
  const originalOverlay = $overlayState.get()
  const overlay = { ...originalOverlay, modelPicker: true }
  $inputSelection.set(selection)
  $overlayState.set(overlay)
  const request = vi.fn(async () => ({ execution_state: 'running', execution_generation: 4 }))

  try {
    await handleDashboardStop({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', sid: 'session-a', epoch: 'epoch-a', generation: 4, nonce: 'a'.repeat(32) }) },
      { gw: { isCanonical: true, request } as unknown as Parameters<typeof handleDashboardStop>[1]['gw'], appendMessage: vi.fn(), sys: vi.fn() })
    expect($inputSelection.get()).toBe(selection)
    expect($overlayState.get()).toEqual(overlay)
    expect(selection.clear).not.toHaveBeenCalled()
  } finally {
    $inputSelection.set(null)
    $overlayState.set(originalOverlay)
  }
})

it('does not apply an acknowledgement after a profile switch', async () => {
  let resolve!: (value: { execution_state: string; execution_generation: number }) => void
  const request = vi.fn(() => new Promise<{ execution_state: string; execution_generation: number }>(r => { resolve = r }))
  patchUiState({ info: { model: 'm', skills: {}, tools: {}, profile_name: 'alpha', execution_epoch: 'epoch-a', execution_generation: 4 } })

  const pending = handleDashboardStop({ type: 'osc', code: 6973, data: JSON.stringify({ action: 'stop', sid: 'session-a', epoch: 'epoch-a', generation: 4, nonce: 'a'.repeat(32) }) },
    { gw: { isCanonical: true, request } as unknown as Parameters<typeof handleDashboardStop>[1]['gw'], appendMessage: vi.fn(), sys: vi.fn() })

  patchUiState({ info: { model: 'm', skills: {}, tools: {}, profile_name: 'beta', execution_epoch: 'epoch-a', execution_generation: 4 } })
  resolve({ execution_state: 'running', execution_generation: 4 })
  await pending
  expect(turnController.interrupted).toBe(false)
})
