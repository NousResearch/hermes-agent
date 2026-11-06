import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const flush = () => new Promise(resolve => setImmediate(resolve))
const pet = { displayName: 'Sabrina', installed: true, slug: 'sabrina' }

function harness(gallery = { active: 'sabrina', enabled: true, pets: [pet] }) {
  const request = vi.fn(async (method: string) => {
    if (method === 'slash.exec') {
      throw new JsonRpcGatewayError('unsupported_command', { code: 4001, data: { reason: 'unsupported_command' } })
    }

    if (method === 'pet.gallery') {
      return gallery
    }

    if (method === 'pet.select') {
      return { ...pet, ok: true }
    }

    if (method === 'pet.scale') {
      return { ok: true, scale: 0.6 }
    }

    return { ok: true }
  })

  const sys = vi.fn()

  const ctx = {
    gateway: { gw: { isCanonical: true, request }, rpc: vi.fn() },
    local: { catalog: null },
    slashFlightRef: { current: 0 },
    transcript: { sys }
  }

  return { handle: createSlashHandler(ctx as never), request, sys }
}

beforeEach(() => {
  resetUiState()
  resetOverlayState()
  patchUiState({ sid: 'owner' })
})

it.each(['/pet', '/pet toggle'])('%s disables the enabled pet without slash.exec', async command => {
  const { handle, request, sys } = harness()
  expect(handle(command)).toBe(true)
  await flush()
  expect(request).toHaveBeenCalledWith('pet.gallery', { localOnly: true })
  expect(request).toHaveBeenCalledWith('pet.disable', {})
  expect(request.mock.calls.some(([method]) => method === 'slash.exec')).toBe(false)
  expect(sys).toHaveBeenCalledWith(expect.stringContaining('put away'))
})

it.each(['/pet', '/pet toggle'])('%s re-enables the selected installed pet', async command => {
  const { handle, request, sys } = harness({ active: 'sabrina', enabled: false, pets: [pet] })
  handle(command)
  await flush()
  expect(request).toHaveBeenCalledWith('pet.select', { slug: 'sabrina' })
  expect(sys).toHaveBeenCalledWith(expect.stringContaining('Sabrina'))
})

it('enables the first installed pet when none is selected', async () => {
  const { handle, request } = harness({ active: '', enabled: false, pets: [pet] })
  handle('/pet')
  await flush()
  expect(request).toHaveBeenCalledWith('pet.select', { slug: 'sabrina' })
})

it('explains how to adopt a pet when none is installed', async () => {
  const { handle, request, sys } = harness({ active: '', enabled: false, pets: [] })
  handle('/pet toggle')
  await flush()
  expect(request.mock.calls.filter(([method]) => method.startsWith('pet.')).map(([method]) => method)).toEqual([
    'pet.gallery'
  ])
  expect(sys).toHaveBeenCalledWith(expect.stringContaining('/pet list'))
})

it.each([
  ['/pet off', 'pet.disable', {}],
  ['/pet scale 0.6', 'pet.scale', { scale: '0.6' }],
  ['/pet sabrina', 'pet.select', { slug: 'sabrina' }]
])('%s calls the existing pet RPC', async (command, method, params) => {
  const { handle, request, sys } = harness()
  handle(command)
  await flush()
  expect(request).toHaveBeenCalledWith(method, params)
  expect(request.mock.calls.some(([name]) => name === 'slash.exec')).toBe(false)
  expect(sys).toHaveBeenCalled()
})

it('keeps the gallery picker local', () => {
  const { handle, request } = harness()
  handle('/pet list')
  expect(getOverlayState().petPicker).toBe(true)
  expect(request.mock.calls.some(([method]) => method.startsWith('pet.'))).toBe(false)
})

it.each(['/pet scale', '/pet off extra', '/pet unknown verb'])('%s reports usage without a mutation', async command => {
  const { handle, request, sys } = harness()
  handle(command)
  await flush()
  expect(request.mock.calls.some(([method]) => method.startsWith('pet.'))).toBe(false)
  expect(sys).toHaveBeenCalledWith(expect.stringContaining('usage: /pet'))
})

it('shows adoption failures with the gallery hint, never a success', async () => {
  const { handle, request, sys } = harness()
  request.mockImplementation(async (method: string) => {
    if (method === 'pet.select') {
      throw new Error("could not adopt 'unknown': unknown pet")
    }

    return { ok: true }
  })
  handle('/pet unknown')
  await flush()
  expect(sys).toHaveBeenCalledWith(expect.stringContaining("could not adopt 'unknown'"))
  expect(sys).toHaveBeenCalledWith(expect.stringContaining('/pet list'))
  expect(request).not.toHaveBeenCalledWith('command.observe', expect.anything())
})

it('does not mutate after a toggle read returns to a different session', async () => {
  const { handle, request, sys } = harness()
  let release!: (value: { active: string; enabled: boolean; pets: (typeof pet)[] }) => void

  const pending = new Promise<{ active: string; enabled: boolean; pets: (typeof pet)[] }>(resolve => {
    release = resolve
  })

  request.mockImplementation(async (method: string) => (method === 'pet.gallery' ? pending : { ok: true }))
  handle('/pet')
  patchUiState({ sid: 'other' })
  release({ active: 'sabrina', enabled: true, pets: [pet] })
  await flush()
  expect(request.mock.calls.some(([method]) => method === 'pet.disable' || method === 'pet.select')).toBe(false)
  expect(sys).not.toHaveBeenCalled()
})

it('reports a successful mutation once to command.observe', async () => {
  const { handle, request } = harness()
  handle('/pet off')
  await flush()
  expect(request).toHaveBeenCalledWith('command.observe', { arg: 'off', name: 'pet', session_id: 'owner' })
})

it.each([
  ['/pet off', 'pet.disable'],
  ['/pet scale bad', 'pet.scale'],
  ['/pet toggle', 'pet.gallery']
])('%s surfaces the RPC failure without reporting success', async (command, failingMethod) => {
  const { handle, request, sys } = harness()
  request.mockImplementation(async (method: string) => {
    if (method === failingMethod) {
      throw new Error('owner refused the operation')
    }

    return { ok: true }
  })
  handle(command)
  await flush()
  expect(sys).toHaveBeenCalledTimes(1)
  expect(sys).toHaveBeenCalledWith(expect.stringContaining('owner refused the operation'))
  expect(request.mock.calls.some(([method]) => method === 'command.observe')).toBe(false)
})

it('does not display a completed mutation in a different session', async () => {
  const { handle, sys } = harness()
  handle('/pet off')
  patchUiState({ sid: 'other' })
  await flush()
  expect(sys).not.toHaveBeenCalled()
})

it('does not re-adopt a missing selected pet when another installed pet is available', async () => {
  const { handle, request } = harness({ active: 'missing', enabled: false, pets: [pet] })
  handle('/pet')
  await flush()
  expect(request).toHaveBeenCalledWith('pet.select', { slug: 'sabrina' })
})
