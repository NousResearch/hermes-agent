import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { turnController } from '../app/turnController.js'
import { $uiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { GatewayClient } from '../gatewayClient.js'

const noop = () => {}

// Frames captured from the real server-side producer (gateway/session_events.py
// SessionEvents.publish) when a subscriber's fanout overflows: the last delta
// that fit, then the gap notice. The subscription is retired server-side, so
// nothing further — not even message.complete — will arrive on it.
const overflowFrames = [
  { jsonrpc: '2.0', method: 'event', params: { authority_epoch: 1, execution_generation: 1, type: 'message.delta',
    session_id: 'owner', payload: { text: 'initial' }, replay_epoch: 'f34cdc57f0ec473da1c1d3a2efb33c46', seq: 1 } },
  { jsonrpc: '2.0', method: 'event', params: { type: 'session.replay_gap', session_id: 'owner',
    payload: { replay_epoch: 'f34cdc57f0ec473da1c1d3a2efb33c46', latest_seq: 258 } } }
]

function mount() {
  const home = mkdtempSync(join(tmpdir(), 'ink-replay-gap-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  patchUiState({ sid: 'owner', busy: true, status: 'running…',
    info: { model: 'test', skills: {}, tools: {}, stored_session_id: 'stored-owner', execution_epoch: '1', execution_generation: 1 } })

  const gw = new GatewayClient()
  gw.isCanonical = true
  const resumeById = vi.fn()

  const onEvent = createGatewayEventHandler({
    composer: { dequeue: () => undefined, queueEditRef: { current: null }, sendQueued: noop, setInput: noop },
    gateway: { gw, rpc: async () => null },
    session: { STARTUP_RESUME_ID: '', colsRef: { current: 80 }, newSession: noop, resetSession: noop, resumeById, setCatalog: noop },
    submission: { submitRef: { current: noop } }, system: { bellOnComplete: false, sys: noop },
    transcript: { appendMessage: noop, panel: noop, setHistoryItems: noop },
    voice: { setProcessing: noop, setRecording: noop, setVoiceEnabled: noop }
  } as any)

  gw.on('event', onEvent)
  gw.drain()

  const instance = renderSync(<Text>probe</Text>, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false
  })

  return {
    resumeById,
    feed(frames: unknown[]) {
      for (const frame of frames) { (gw as any).handleWebSocketFrame(JSON.stringify(frame)) }
    },
    cleanup() { instance.unmount(); turnController.fullReset(); resetUiState(); vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) }
  }
}

it('re-attaches the current session when its subscription overflows on a healthy socket', async () => {
  const h = mount()

  try {
    await new Promise(resolve => setImmediate(resolve))
    h.feed(overflowFrames)
    await expect.poll(() => h.resumeById.mock.calls).toEqual([['owner']])
  } finally { h.cleanup() }
})

it('ignores a replay gap for a session that is not in focus', async () => {
  const h = mount()

  try {
    await new Promise(resolve => setImmediate(resolve))
    h.feed([{ ...overflowFrames[1], params: { ...overflowFrames[1]!.params, session_id: 'elsewhere' } }])
    await new Promise(resolve => setImmediate(resolve))
    expect(h.resumeById).not.toHaveBeenCalled()
    expect($uiState.get().sid).toBe('owner')
  } finally { h.cleanup() }
})
