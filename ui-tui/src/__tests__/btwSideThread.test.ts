import { beforeEach, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// /btw side exploration threads: the gateway parses the flags (agent/side_thread.py) and Ink only
// renders the reply, submits a promotion as an ordinary turn and prints btw.progress steps.

const flush = () => new Promise(resolve => setImmediate(resolve))

function harness(reply: Record<string, string>) {
  const request = vi.fn(async () => reply)
  const sys = vi.fn()
  const send = vi.fn()
  const enqueue = vi.fn()

  const ctx = {
    slashFlightRef: { current: 0 },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    transcript: { page: vi.fn(), panel: vi.fn(), send, sys, setHistoryItems: vi.fn(), trimLastExchange: (items: unknown[]) => items },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    composer: { enqueue }
  }

  return { enqueue, request, send, slash: createSlashHandler(ctx as any), sys }
}

beforeEach(() => {
  resetUiState()
  patchUiState({ sid: 'owner', info: { model: 'm', skills: {}, tools: {}, execution_epoch: '1', execution_generation: 3 } as any })
})

it('/btw --explore shows the gateway start notice instead of the snapshot one', async () => {
  const { request, slash, sys } = harness({ task_id: 'btw_ab12cd', text: '🔭 Side exploration btw_ab12cd: "find it"' })

  slash('/btw --explore find it')
  await flush()

  expect(request).toHaveBeenCalledWith('prompt.btw', { session_id: 'owner', text: '--explore find it' })
  expect(sys.mock.calls.flat()).toEqual(['🔭 Side exploration btw_ab12cd: "find it"'])
})

it('/btw --promote submits the returned prompt as a turn when idle', async () => {
  const { enqueue, send, slash, sys } = harness({ text: '⤴ Bringing btw_ab12cd in', prompt: 'PROMOTED' })

  slash('/btw --promote btw_ab12cd')
  await flush()

  expect(sys).toHaveBeenCalledWith('⤴ Bringing btw_ab12cd in')
  expect(send).toHaveBeenCalledWith('PROMOTED', true, '/btw --promote btw_ab12cd')
  expect(enqueue).not.toHaveBeenCalled()
})

it('/btw --promote queues the prompt behind a running turn', async () => {
  patchUiState({ busy: true })
  const { enqueue, send, slash } = harness({ text: '⤴', prompt: 'PROMOTED' })

  slash('/btw --promote')
  await flush()

  expect(enqueue).toHaveBeenCalledWith('PROMOTED')
  expect(send).not.toHaveBeenCalled()
})

it('btw.progress prints the step under its thread id', () => {
  const sys = vi.fn()
  const ref = <T>(current: T) => ({ current })

  const onEvent = createGatewayEventHandler({
    composer: { dequeue: () => undefined, queueEditRef: ref(null), sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: { STARTUP_RESUME_ID: '', colsRef: ref(80), newSession: vi.fn(), resetSession: vi.fn(), resumeById: vi.fn(), setCatalog: vi.fn() },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  } as any)

  onEvent({ payload: { task_id: 'btw_ab12cd', text: 'search_files · retry' }, session_id: 'owner', type: 'btw.progress' } as any)

  expect(sys).toHaveBeenCalledWith('[btw btw_ab12cd] ┊ search_files · retry')
})
