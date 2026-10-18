import { PassThrough } from 'node:stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { applyAgentSnapshot } from '../app/agentRoster.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import { requestLiveSessions } from '../app/liveSessions.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { sendAgentSteer } from '../components/agentControls.js'
import { AgentsOverlay } from '../components/agentsOverlay.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'

// Ink verbs that resolve their session_id in the legacy sidecar's own map. On the shared gateway the
// sidecar never holds the session, so each answered "session not found" (or an empty roster). Every
// site here must either use what the owner serves or refuse by name, never reach the sidecar verb.
// The owner serves subagent.steer and subagent.interrupt itself (gateway/session_ancillary.py).
const SIDECAR_SESSION_VERBS = ['session.active_list']

const owner = (answer: (method: string) => unknown = () => ({})) => {
  const request = vi.fn(async (method: string, _params?: unknown) => {
    if (SIDECAR_SESSION_VERBS.includes(method)) {
      throw new Error('session not found')
    }

    return answer(method)
  })

  return { gw: { isCanonical: true, request } as unknown as GatewayClient, request }
}

const sidecarCalls = (request: ReturnType<typeof vi.fn>) =>
  request.mock.calls.map(([method]) => method as string).filter(method => SIDECAR_SESSION_VERBS.includes(method))

afterEach(() => resetUiState())

async function overlayKey(
  gw: GatewayClient,
  key: string,
  expected = '/agents kill is not available on the shared gateway yet'
) {
  patchUiState({ sid: 'owner' })
  applyAgentSnapshot('owner', {
    delegations: [],
    subagents: [{ depth: 0, goal: 'child', status: 'running', subagent_id: 'c1' } as never]
  })
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 24, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let frame = ''
  stdout.on('data', chunk => (frame += String(chunk)))

  const view = renderSync(
    <Box height={24}>
      <AgentsOverlay gw={gw} onClose={() => {}} t={DEFAULT_THEME} />
    </Box>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  try {
    await vi.waitFor(() => expect(frame).toContain('child'))
    stdin.write(key)
    await vi.waitFor(() => expect(frame).toContain(expected))
  } finally {
    view.unmount()
    view.cleanup()
    applyAgentSnapshot(null)
  }
}

const SITES: [string, () => Promise<void>][] = [
  [
    "session.ts::/voice tts (the owner's /voice handler holds the session state)",
    async () => {
      patchUiState({ sid: 'owner' })
      const { gw, request } = owner()
      const sys = vi.fn()

      createSlashHandler({
        gateway: { gw, rpc: request },
        local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
        session: {},
        slashFlightRef: { current: 0 },
        transcript: { page: vi.fn(), panel: vi.fn(), send: vi.fn(), setHistoryItems: vi.fn(), sys },
        voice: { setVoiceEnabled: vi.fn(), setVoiceRecordKey: vi.fn(), setVoiceTts: vi.fn() }
      } as never)('/voice tts')
      await new Promise(resolve => setImmediate(resolve))
      // The session's voice state is the owner's: `/voice tts` runs its `/voice` handler, never the
      // sidecar's voice.toggle.
      expect(request).toHaveBeenCalledWith('slash.exec', { command: 'voice tts', session_id: 'owner' })
      expect(request.mock.calls.some(([method]) => method === 'voice.toggle')).toBe(false)
    }
  ],
  [
    "agentControls::sendAgentSteer is the owner's subagent.steer",
    async () => {
      const { gw, request } = owner(method => (method === 'subagent.steer' ? { status: 'queued' } : {}))
      const result = await sendAgentSteer(gw, 'owner', 'c1', 'check tests')
      expect(result.accepted).toBe(true)
      expect(request).toHaveBeenCalledWith('subagent.steer', {
        session_id: 'owner',
        subagent_id: 'c1',
        text: 'check tests'
      })
      expect(sidecarCalls(request)).toEqual([])
    }
  ],
  [
    "agentsOverlay::killOne (x) is the owner's subagent.interrupt",
    async () => {
      const { gw, request } = owner(method => (method === 'subagent.interrupt' ? { found: true } : {}))
      await overlayKey(gw, 'x', 'c1')
      await vi.waitFor(() =>
        expect(request).toHaveBeenCalledWith('subagent.interrupt', { session_id: 'owner', subagent_id: 'c1' })
      )
      expect(sidecarCalls(request)).toEqual([])
    }
  ],
  [
    "agentsOverlay::killSubtree (X) is the owner's subagent.interrupt",
    async () => {
      const { gw, request } = owner(method => (method === 'subagent.interrupt' ? { found: true } : {}))
      await overlayKey(gw, 'X', 'killing')
      expect(request).toHaveBeenCalledWith('subagent.interrupt', { session_id: 'owner', subagent_id: 'c1' })
      expect(sidecarCalls(request)).toEqual([])
    }
  ],
  [
    "agentsOverlay::togglePause (p) is the owner's delegation.pause",
    async () => {
      const { gw, request } = owner(method => (method === 'delegation.pause' ? { paused: true } : {}))
      await overlayKey(gw, 'p', 'spawning paused')
      expect(request).toHaveBeenCalledWith('delegation.pause', { paused: true })
      expect(sidecarCalls(request)).toEqual([])
    }
  ],
  [
    'useMainApp::live-session poll (status-bar title + count)',
    async () => {
      const { gw, request } = owner(() => ({ sessions: [{ id: 'owner', title: 'Refactor plan' }] }))
      const result = (await requestLiveSessions(gw, 'owner')) as { sessions: { title: string }[] }
      expect(result.sessions[0]?.title).toBe('Refactor plan')
      expect(request).toHaveBeenCalledWith('session.list', { limit: 200 })
      expect(sidecarCalls(request)).toEqual([])
    }
  ]
]

it.each(SITES)('%s never reaches a sidecar session verb on the shared gateway', async (_site, run) => {
  await run()
})
