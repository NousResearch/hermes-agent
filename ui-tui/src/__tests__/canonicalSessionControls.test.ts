import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

const flush = () => new Promise(resolve => setImmediate(resolve))
const info = { model: 'old', skills: {}, tools: {}, execution_epoch: '1', execution_generation: 4 }

function harness() {
  let compressed = false

  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.resume') {
      return {
        session_id: 'owner',
        revision: 8,
        execution_generation: 4,
        info: { ...info, execution_generation: compressed ? 5 : 4 },
        messages: [{ role: 'assistant', text: 'summary' }]
      }
    }

    if (method === 'session.mutate') {
      compressed = params.operation === 'compress'

      return { model: 'picked', branched_session_id: 'child', execution_generation: 5 }
    }

    return { value: params.value ?? 'queue', status: 'queued' }
  })

  const sys = vi.fn()

  const ctx = {
    slashFlightRef: { current: 0 },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    transcript: { sys, setHistoryItems: vi.fn() },
    local: { maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    composer: { enqueue: vi.fn() }
  }

  return { ctx, request, slash: createSlashHandler(ctx as any) }
}

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner', info })
})

it('canonical model, branch and compression use generation-fenced owner mutations and hydrate results', async () => {
  const { ctx, request, slash } = harness()

  for (const [command, operation, payload] of [
    ['/model picked --provider custom --tui-session', 'model', { model: 'picked', provider: 'custom' }],
    ['/branch Child name', 'branch', { title: 'Child name' }],
    ['/compress keep context', 'compress', { focus: 'keep context' }]
  ] as const) {
    slash(command)
    await flush()
    expect(request).toHaveBeenCalledWith('session.mutate', {
      session_id: 'owner',
      request_id: expect.any(String),
      expected_revision: 8,
      expected_generation: 4,
      operation,
      payload
    })

    if (operation === 'model') {
      expect(getUiState().info?.execution_generation).toBe(5)
    }
  }

  expect(ctx.session.resumeById).toHaveBeenCalledWith('child')
  expect(ctx.transcript.setHistoryItems).toHaveBeenCalledWith(
    expect.arrayContaining([expect.objectContaining({ role: 'assistant', text: 'summary' })])
  )
  expect(
    request.mock.calls.some(([method]) => ['config.set', 'session.branch', 'session.compress'].includes(method))
  ).toBe(false)
})

it('ambiguous mutations retry the frozen identity while unsupported model scopes never mutate', async () => {
  const { request, slash } = harness()
  request.mockImplementation(async (method: string) => {
    if (method === 'session.resume') {
      return { revision: 8, execution_generation: 4 } as any
    }

    throw new Error('connection lost')
  })
  slash('/branch same')
  await flush()
  slash('/branch same')
  await flush()
  const calls = request.mock.calls.filter(([method]) => method === 'session.mutate')
  expect(calls).toHaveLength(2)
  expect(calls[0]![1]).toEqual(calls[1]![1])
  slash('/model picked --global')
  await flush()
  expect(request.mock.calls.filter(([method]) => method === 'session.mutate')).toHaveLength(2)
})

it('late model or compression replies cannot replace a newer execution snapshot', async () => {
  for (const command of ['/model picked', '/compress']) {
    resetUiState()
    patchUiState({ sid: 'owner', info })
    const { ctx, request, slash } = harness()
    let release!: (value: any) => void
    request.mockImplementation(async (method: string) => {
      if (method === 'session.resume') {
        return { revision: 8, execution_generation: 4, info, messages: [] } as any
      }

      return new Promise(resolve => {
        release = resolve
      })
    })
    slash(command)
    await flush()
    patchUiState({ info: { ...info, model: 'newer', execution_generation: 6 } })
    release({ model: 'picked', execution_generation: 5 })
    await flush()
    expect(getUiState().info).toMatchObject({ model: 'newer', execution_generation: 6 })
    expect(ctx.transcript.setHistoryItems).not.toHaveBeenCalled()
  }
})

it('busy mode is session scoped and steering includes the observed execution generation', async () => {
  patchUiState({ busy: true })
  const { request, slash } = harness()
  slash('/busy steer')
  await flush()
  expect(request).toHaveBeenCalledWith('config.set', { key: 'busy', value: 'steer', session_id: 'owner' })
  expect(getUiState().busyInputMode).toBe('steer')
  slash('/busy status')
  await flush()
  expect(request).toHaveBeenCalledWith('config.get', { key: 'busy', session_id: 'owner' })
  slash('/steer correction')
  await flush()
  expect(request).toHaveBeenCalledWith('session.steer', {
    session_id: 'owner',
    text: 'correction',
    execution_generation: 4
  })
})
