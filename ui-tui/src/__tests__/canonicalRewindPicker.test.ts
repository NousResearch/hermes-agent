import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { $overlayState, resetOverlayState } from '../app/overlayStore.js'
import { rewindTargets } from '../app/slash/canonicalRewindPicker.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { shouldOpenRewindOnDoubleEsc } from '../app/useInputHandlers.js'

const flush = () => new Promise(resolve => setImmediate(resolve))

const messages = [
  { role: 'user', content: 'first', row_id: 11 },
  { role: 'assistant', content: 'a1', row_id: 12 },
  { role: 'user', content: [{ type: 'input_image', image_url: 'x' }], row_id: 13 },
  { role: 'assistant', content: 'a2', row_id: 14 },
  { role: 'user', content: 'third question', row_id: 15 },
  { role: 'assistant', content: 'a3', row_id: 16 }
]

function harness(rewindRefusal?: Error, repaintFailure?: Error) {
  let cut = false

  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.resume') {
      if (cut && repaintFailure) {
        throw repaintFailure
      }

      return { session_id: 'owner', revision: 7, execution_generation: 3, messages, info: { model: 'm' } }
    }

    if (method === 'session.mutate' && params.operation === 'rewind') {
      if (rewindRefusal) {
        throw rewindRefusal
      }

      cut = true

      return { session_id: 'owner', operation: 'rewind', revision: 8, execution_generation: 4, rewound_count: 6 }
    }

    throw new Error(`unknown_method: ${method}`)
  })

  const sys = vi.fn()
  const send = vi.fn()
  const setHistoryItems = vi.fn()
  const composer = { enqueue: vi.fn(), setInput: vi.fn() }

  const ctx = {
    slashFlightRef: { current: 0 },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    transcript: { page: vi.fn(), panel: vi.fn(), send, sys, setHistoryItems, trimLastExchange: (items: unknown[]) => items },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    composer,
    voice: {}
  }

  return { composer, request, send, setHistoryItems, slash: createSlashHandler(ctx as any), sys }
}

const mutations = (request: ReturnType<typeof harness>['request']) =>
  request.mock.calls.filter(([method]) => method === 'session.mutate')

beforeEach(() => {
  resetUiState()
  resetOverlayState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {}, execution_epoch: '1', execution_generation: 3 } as any })
})

describe('/rewind picker (#21910)', () => {
  it('offers every earlier text prompt, numbered by turn; a media turn keeps its number but is not offered', () => {
    expect(rewindTargets({ messages })).toEqual([
      { rowId: 11, text: 'first', turn: 1 },
      { rowId: 15, text: 'third question', turn: 3 }
    ])
  })

  it('cuts nothing until the edit is sent, then rewinds to the PICKED row and sends the edited text', async () => {
    const { request, send, setHistoryItems, slash } = harness()

    slash('/rewind')
    await flush()

    const picker = $overlayState.get().rewind

    expect(picker?.targets.map(target => target.rowId)).toEqual([11, 15])
    expect(mutations(request)).toHaveLength(0)

    picker!.onSubmit(picker!.targets[0]!, 'first, but in French')
    await flush()
    await flush()

    expect(mutations(request)).toHaveLength(1)
    expect(mutations(request)[0]![1]).toMatchObject({
      expected_revision: 7,
      expected_generation: 3,
      operation: 'rewind',
      payload: { target_message_id: 11 }
    })
    expect(setHistoryItems).toHaveBeenCalled()
    expect(send).toHaveBeenCalledWith('first, but in French')
  })

  it('a busy refusal at send time prints the stop-first text and sends nothing', async () => {
    const busy = new JsonRpcGatewayError('session_busy', { code: 4001, data: { reason: 'session_busy' } })
    const { composer, send, slash, sys } = harness(busy)

    slash('/rewind')
    await flush()
    $overlayState.get().rewind!.onSubmit($overlayState.get().rewind!.targets[1]!, 'again')
    await flush()
    await flush()

    expect(send).not.toHaveBeenCalled()
    expect(sys.mock.calls.map(([text]) => String(text)).join('\n')).toContain('Stop the current reply first')
    // Nothing was cut, so the edit goes back into the composer instead of vanishing with the picker.
    expect(composer.setInput).toHaveBeenCalledWith('again')
  })

  it('a cut that commits but cannot repaint keeps the edit as that session\'s next prompt', async () => {
    const { composer, send, slash } = harness(undefined, new Error('transport closed'))

    slash('/rewind')
    await flush()
    $overlayState.get().rewind!.onSubmit($overlayState.get().rewind!.targets[0]!, 'first, edited')
    await flush()
    await flush()
    await flush()

    expect(send).not.toHaveBeenCalled()
    expect(composer.enqueue).toHaveBeenCalledWith('first, edited', 'first, edited', expect.objectContaining({ sid: 'owner' }))
  })

  it('does not open while a turn runs', async () => {
    patchUiState({ busy: true })
    const { request, slash } = harness()

    slash('/rewind')
    await flush()

    expect($overlayState.get().rewind).toBeNull()
    expect(request.mock.calls.some(([method]) => method === 'session.resume' || method === 'session.mutate')).toBe(false)
  })
})

describe('idle double-Esc opens the rewind picker', () => {
  const idle = { blocked: false, busy: false, canonical: true, hasDraft: false, hasSession: true }

  it('only on an idle, empty composer with no overlay', () => {
    expect(shouldOpenRewindOnDoubleEsc(idle)).toBe(true)
    expect(shouldOpenRewindOnDoubleEsc({ ...idle, busy: true })).toBe(false)
    expect(shouldOpenRewindOnDoubleEsc({ ...idle, hasDraft: true })).toBe(false)
    expect(shouldOpenRewindOnDoubleEsc({ ...idle, blocked: true })).toBe(false)
    expect(shouldOpenRewindOnDoubleEsc({ ...idle, hasSession: false })).toBe(false)
    expect(shouldOpenRewindOnDoubleEsc({ ...idle, canonical: false })).toBe(false)
  })
})
