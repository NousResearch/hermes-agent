import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { $overlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { $uiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { GatewayClient } from '../gatewayClient.js'

const noop = () => {}

// Canonical attach snapshot: `prompts` carries the controls still waiting on
// this session (approval / clarify) so a viewer can hydrate them on attach.
const snapshotA = {
  session_id: 'a', stored_session_id: 'a', subscription_id: 'sub-a', authority_epoch: 1, execution_generation: 4,
  running: true, status: 'waiting', messages: [],
  info: { model: 'test', tools: {}, skills: {}, stored_session_id: 'a', execution_epoch: '1', execution_generation: 4 },
  prompts: [{ kind: 'approval', prompt_id: 'p-a', execution_generation: 4, description: 'review command',
    command: 'sentinel', choices: ['once', 'deny'] }]
}

const snapshotB = { ...snapshotA, session_id: 'b', stored_session_id: 'b', subscription_id: 'sub-b', running: false,
  prompts: [], info: { ...snapshotA.info, stored_session_id: 'b' } }

function mount() {
  const home = mkdtempSync(join(tmpdir(), 'ink-activate-prompts-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()

  const gw = new GatewayClient()
  gw.isCanonical = true

  const onEvent = createGatewayEventHandler({
    composer: { dequeue: () => undefined, queueEditRef: { current: null }, sendQueued: noop, setInput: noop },
    gateway: { gw, rpc: async () => null },
    session: { STARTUP_RESUME_ID: '', colsRef: { current: 80 }, newSession: noop, resetSession: noop, resumeById: noop, setCatalog: noop },
    submission: { submitRef: { current: noop } }, system: { bellOnComplete: false, sys: noop },
    transcript: { appendMessage: noop, panel: noop, setHistoryItems: noop },
    voice: { setProcessing: noop, setRecording: noop, setVoiceEnabled: noop }
  } as any)

  gw.on('event', onEvent)
  gw.drain()

  vi.spyOn(gw, 'request').mockImplementation(async (method, params) =>
    method === 'session.detach' ? {} : (params as { session_id?: string })?.session_id === 'b' ? snapshotB : snapshotA)

  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 }, composerActions: { setComposerTokens: noop }, gw, rpc: async () => null,
      scrollRef: { current: null }, panel: noop, sys: noop, setHistoryItems: noop, setLastUserMsg: noop,
      setSessionStartedAt: noop, setStickyPrompt: noop, setVoiceProcessing: noop, setVoiceRecording: noop
    } as any)

    return <Text>probe</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false
  })

  return {
    get lifecycle() { return lifecycle },
    cleanup() { instance.unmount(); turnController.fullReset(); resetUiState(); vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) }
  }
}

it('activating a live session hydrates its pending approval exactly like resume does', async () => {
  const h = mount()

  try {
    await new Promise(resolve => setImmediate(resolve))
    h.lifecycle.resumeById('a')
    await expect.poll(() => $overlayState.get().approval?.sharedControl?.prompt_id).toBe('p-a')

    h.lifecycle.activateLiveSession('b')
    await expect.poll(() => $uiState.get().sid).toBe('b')
    expect($overlayState.get().approval).toBeNull()

    h.lifecycle.activateLiveSession('a')
    await expect.poll(() => $uiState.get().sid).toBe('a')
    await expect.poll(() => $overlayState.get().approval?.sharedControl?.prompt_id).toBe('p-a')
  } finally { h.cleanup() }
})
