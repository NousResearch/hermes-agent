import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { modelPickerCommand } from '../components/modelPicker.js'

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

    if (method === 'session.info') {
      return { title: params.session_id === 'child' ? 'Parent #2' : 'Parent' }
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

it.each(['high', 'none'])(
  'applies picker reasoning %s after the model switch through existing controls',
  async effort => {
    const { request, slash } = harness()
    let generation = 4
    request.mockImplementation(async (method: string, params: any) => {
      if (method === 'session.resume') {
        return { revision: generation + 4, execution_generation: generation } as any
      }

      if (method === 'session.mutate') {
        generation += 1

        return { model: 'picked', effort: params.payload.effort, execution_generation: generation }
      }

      return {}
    })

    slash(`/model ${modelPickerCommand('picked', 'custom', false, effort)}`)
    await flush()

    expect(request).toHaveBeenCalledWith('session.mutate', {
      session_id: 'owner',
      request_id: expect.any(String),
      expected_revision: 8,
      expected_generation: 4,
      operation: 'model',
      payload: { model: 'picked', provider: 'custom' }
    })
    expect(request).toHaveBeenCalledWith('session.mutate', {
      session_id: 'owner',
      request_id: expect.any(String),
      expected_revision: 9,
      expected_generation: 5,
      operation: 'reasoning',
      payload: { effort }
    })
    expect(getUiState().info?.model).toBe('picked')
  }
)

it('does not apply picker reasoning after the model generation has been superseded', async () => {
  const { request, slash } = harness()
  let reads = 0
  request.mockImplementation(async (method: string) => {
    if (method === 'session.resume') {
      reads += 1

      return { revision: 8 + reads, execution_generation: reads === 1 ? 4 : 6 } as any
    }

    return { model: 'picked', execution_generation: 5 }
  })

  slash(`/model ${modelPickerCommand('picked', 'custom', false, 'high')}`)
  await flush()

  const sends = request.mock.calls.filter(([method]) => method === 'session.mutate')
  expect(sends).toHaveLength(1)
  expect(sends[0]![1].operation).toBe('model')
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

it('a guarded canonical model switch asks; yes re-sends once with the owner token, no changes nothing', async () => {
  for (const accept of [false, true]) {
    resetUiState()
    resetOverlayState()
    patchUiState({ sid: 'owner', info })
    const { request, slash } = harness()
    let generation = 4
    request.mockImplementation(async (method: string, params: any) => {
      if (method === 'session.resume') {
        return { revision: generation + 4, execution_generation: generation, info } as any
      }

      if (method !== 'session.mutate') {
        return {}
      }

      if (params.operation === 'reasoning') {
        return { effort: params.payload.effort, execution_generation: 6 }
      }

      if (params.payload.confirm === 'tok-1') {
        generation = 5

        return { model: 'pricey', execution_generation: generation }
      }

      return {
        status: 'confirmation_required',
        confirm: 'tok-1',
        confirm_message: 'pricey IS EXPENSIVE',
        target_model: 'pricey'
      }
    })
    slash(`/model ${modelPickerCommand('pricey', 'custom', false, 'high')}`)
    await flush()
    const ask = getOverlayState().confirm
    expect(ask?.detail).toBe('pricey IS EXPENSIVE')
    expect(getUiState().info?.model).toBe('old')

    if (accept) {
      ask!.onConfirm()
      await flush()
      await flush()
    }

    const sends = request.mock.calls
      .filter(([method]) => method === 'session.mutate')
      .map(([, params]) => params.payload)

    const selection = { model: 'pricey', provider: 'custom' }
    expect(sends).toEqual(accept ? [selection, { ...selection, confirm: 'tok-1' }, { effort: 'high' }] : [selection])
    expect(getUiState().info?.model).toBe(accept ? 'pricey' : 'old')
  }
})

it('/branch prints main\'s "branched →" confirmation after the child resume has repainted', async () => {
  // The resume resets the visible history: a line printed before it settles is wiped (FINDINGS row 26).
  const { ctx, slash } = harness()
  const order: string[] = []
  let settle!: () => void
  ctx.session.resumeById = vi.fn(() => {
    order.push('resume')

    return new Promise<void>(resolve => {
      settle = () => {
        order.push('repainted')
        patchUiState({ sid: 'child' }) // the resume moves the view (and the slash's destination) to the child
        resolve()
      }
    })
  })
  ctx.transcript.sys = vi.fn((line: string) => order.push(`sys:${line}`))

  slash('/branch Child name')
  await flush()
  expect(order).toEqual(['resume'])
  settle()
  await flush()
  expect(order).toEqual(['resume', 'repainted', 'sys:branched → Parent #2'])
})

it('a /compress that removes nothing prints main\'s "No changes" report without re-hydrating', async () => {
  const { ctx, request, slash } = harness()
  request.mockImplementation(async (method: string) => {
    if (method === 'session.resume') {
      return { session_id: 'owner', revision: 8, execution_generation: 4, info, messages: [] } as any
    }

    return {
      session_id: 'owner',
      operation: 'compress',
      status: 'noop',
      summary: {
        headline: 'No changes from compression: 2 messages',
        noop: true,
        token_line: 'Approx request size: ~6,547 tokens (unchanged)'
      }
    } as any
  })

  slash('/compress')
  await flush()
  await flush()
  expect(ctx.transcript.sys).toHaveBeenCalledWith('No changes from compression: 2 messages')
  expect(ctx.transcript.sys).toHaveBeenCalledWith('  Approx request size: ~6,547 tokens (unchanged)')
  expect(ctx.transcript.setHistoryItems).not.toHaveBeenCalled()
  expect(request.mock.calls.filter(([method]) => method === 'session.resume')).toHaveLength(1)
})
