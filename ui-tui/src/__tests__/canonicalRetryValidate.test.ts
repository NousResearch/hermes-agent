import { beforeEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// dokterdok N92 (Ink half): /retry validates the target turn BEFORE rewinding it, as the classic chat
// and main's retry_last do (agent/context_compressor_retry.py::retryable_user_text): an image turn is
// refused with nothing archived; a text-parts turn is resubmitted as its own text, never as whatever
// this view typed last.

const flush = () => new Promise(resolve => setImmediate(resolve))

function harness(content: unknown) {
  const messages = [
    { role: 'user', content: 'first', row_id: 11 },
    { role: 'assistant', content: 'a1', row_id: 12 },
    { role: 'user', content, row_id: 13 },
    { role: 'assistant', content: 'a2', row_id: 14 }
  ]

  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.resume') {
      return { session_id: 'owner', revision: 7, execution_generation: 3, messages }
    }

    if (method === 'session.mutate' && params.operation === 'rewind') {
      return {
        session_id: 'owner',
        operation: 'rewind',
        revision: 8,
        execution_generation: 4,
        rewound_count: 2,
        target_message: { role: 'user', content }
      }
    }

    throw new Error(`unknown_method: ${method}`)
  })

  const sys = vi.fn()
  const send = vi.fn()

  const ctx = {
    slashFlightRef: { current: 0 },
    gateway: { gw: { isCanonical: true, request }, rpc: request },
    transcript: {
      page: vi.fn(),
      panel: vi.fn(),
      send,
      sys,
      setHistoryItems: vi.fn(),
      trimLastExchange: (i: unknown[]) => i
    },
    local: { getHistoryItems: () => messages, getLastUserMsg: () => 'typed elsewhere', maybeWarn: vi.fn() },
    session: { resumeById: vi.fn() },
    composer: { enqueue: vi.fn() }
  }

  return { request, send, slash: createSlashHandler(ctx as any), sys }
}

beforeEach(() => {
  resetUiState()
  patchUiState({
    sid: 'owner',
    info: { model: 'm', skills: {}, tools: {}, execution_epoch: '1', execution_generation: 3 } as any
  })
})

it('/retry refuses an image turn before any rewind and resubmits a text-parts turn as its text', async () => {
  const image = harness([
    { type: 'text', text: 'what is this?' },
    { type: 'image_url', image_url: { url: 'data:x' } }
  ])

  image.slash('/retry')
  await flush()
  expect(image.request.mock.calls.some(([method]) => method === 'session.mutate')).toBe(false)
  expect(image.send).not.toHaveBeenCalled()
  expect(image.sys).toHaveBeenCalledWith(expect.stringContaining('retry does not support media'))

  const parts = harness([
    { type: 'text', text: 'what is ' },
    { type: 'input_text', text: 'this?' }
  ])

  parts.slash('/retry')
  await flush()
  expect(parts.request).toHaveBeenCalledWith(
    'session.mutate',
    expect.objectContaining({ payload: { target_message_id: 13 } })
  )
  expect(parts.send).toHaveBeenCalledWith('what is this?')
})
