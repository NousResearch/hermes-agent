import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { handoffCommands } from '../app/slash/commands/handoff.js'
import { submitPrompt } from '../app/submissionCore.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'

const flush = () => new Promise<void>(resolve => setImmediate(resolve))

const deferred = () => {
  let resolve!: (value: any) => void

  const promise = new Promise<any>(done => {
    resolve = done
  })

  return { promise, resolve }
}

let cleanup: () => void

beforeEach(() => {
  resetUiState()
  turnController.fullReset()
  patchUiState({ sid: 'source', busy: false })
  vi.stubEnv('HERMES_TUI_ACTIVE_SESSION_FILE', '')
})
afterEach(() => {
  cleanup?.()
  vi.unstubAllEnvs()
})

async function mount() {
  const ack = deferred()

  const request = vi.fn((method: string) => {
    if (method === 'handoff.request') {
      return ack.promise
    }

    if (method === 'handoff.state') {
      return Promise.resolve({ state: 'completed' })
    }

    return Promise.resolve({ session_id: 'source', messages: [], provider_configured: true })
  })

  const sys = vi.fn()
  const gw = { request } as any
  let session!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    session = useSessionLifecycle({
      gw,
      rpc: request as any,
      sys,
      colsRef: { current: 80 },
      scrollRef: { current: null },
      composerActions: { setComposerTokens: vi.fn() } as any,
      panel: vi.fn(),
      setHistoryItems: vi.fn(),
      setLastUserMsg: vi.fn(),
      setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn()
    })

    return null
  }

  const stream = new PassThrough()

  const instance = renderSync(React.createElement(Harness), {
    stdout: stream as unknown as NodeJS.WriteStream,
    stderr: stream as unknown as NodeJS.WriteStream,
    stdin: new PassThrough() as unknown as NodeJS.ReadStream,
    patchConsole: false
  })

  cleanup = () => {
    instance.unmount()
    instance.cleanup()
  }

  await flush()

  const handoff = () =>
    handoffCommands[0]!.run(
      'slack',
      {
        sid: getUiState().sid,
        ui: getUiState(),
        gateway: { gw },
        composer: { queueRef: { current: [] } },
        transcript: { sys }
      } as any,
      'handoff'
    )

  const submit = () =>
    submitPrompt('ordinary prompt', {
      gw,
      sys,
      appendMessage: vi.fn(),
      enqueue: vi.fn(),
      expand: text => text,
      setLastUserMsg: vi.fn()
    })

  return { session, request, ack, handoff, submit }
}

it.each(
  (['activate', 'resume', 'resume-setup', 'new-setup', 'new-create'] as const).flatMap(stage =>
    [false, true].map(settled => ({ stage, settled }))
  )
)('discards a pre-handoff $stage continuation (settled=$settled)', async ({ stage, settled }) => {
  const h = await mount()
  const late = deferred()

  const method = {
    activate: 'session.activate',
    resume: 'session.resume',
    'resume-setup': 'setup.status',
    'new-setup': 'setup.status',
    'new-create': 'session.create'
  }[stage]

  const original = h.request.getMockImplementation()!
  h.request.mockImplementation(name => (name === method ? late.promise : original(name)))
  let newSession: Promise<unknown> | undefined

  if (stage === 'activate') {
    h.session.activateLiveSession('source')
  } else if (stage.startsWith('resume')) {
    h.session.resumeById('other')
  } else {
    newSession = h.session.newLiveSession()
  }

  await flush()
  h.handoff()
  const callsAtHandoff = h.request.mock.calls.length

  // Invalidate before the response, then settle: checking only "pending"
  // at callback time would still let an old response reclaim the source.
  if (settled) {
    h.ack.resolve({ queued: true, platform: 'slack', home_name: 'test' })
    await flush()
  }

  late.resolve({ session_id: stage.startsWith('resume') ? 'other' : 'source', messages: [], provider_configured: true })
  await newSession
  await flush()
  h.submit()
  expect(getUiState()).toMatchObject({
    sid: null,
    handoffSessionId: settled ? null : 'source',
    status: settled ? 'handoff completed' : 'handoff pending…'
  })
  expect(h.request.mock.calls.slice(callsAtHandoff).map(([name]) => name)).toEqual(settled ? ['handoff.state'] : [])

  if (!settled) {
    h.ack.resolve({ queued: true, platform: 'slack', home_name: 'test' })
    await flush()
  }
})

it('blocks picker lifecycle mutations and prompts until the handoff settles', async () => {
  const h = await mount()
  h.handoff()
  h.session.activateLiveSession('source')
  await flush()
  h.submit()
  h.session.resumeById('source')
  await h.session.closeSession('source')
  await h.session.newLiveSession()
  await h.session.newSession()
  await flush()
  expect(h.request.mock.calls.map(([method]) => method)).toEqual(['handoff.request'])
  expect(getUiState()).toMatchObject({ sid: null, handoffSessionId: 'source' })

  h.ack.resolve({ queued: true, platform: 'slack', home_name: 'test' })
  await flush()
  expect(getUiState().handoffSessionId).toBeNull()
  h.session.activateLiveSession('source')
  await flush()
  expect(getUiState().sid).toBe('source')
  h.session.resumeById('source')
  await flush()
  await h.session.closeSession('source')
  h.submit()
  await flush()

  for (const method of ['session.activate', 'session.resume', 'session.close', 'prompt.submit']) {
    expect(h.request).toHaveBeenCalledWith(method, expect.objectContaining({ session_id: 'source' }))
  }
})
