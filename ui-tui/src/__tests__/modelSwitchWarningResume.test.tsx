import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import type { Msg } from '../types.js'

it('keeps a deferred warning on resume without duplicating its stored row or unrelated system rows', async () => {
  resetUiState()
  const history: Msg[] = []

  const setHistoryItems = (update: Msg[] | ((previous: Msg[]) => Msg[])) => {
    history.splice(0, history.length, ...(typeof update === 'function' ? update(history) : update))
  }

  const request = vi.fn(async (_method: string, params: { session_id: string }) => ({
    session_id: params.session_id,
    stored_session_id: params.session_id,
    messages:
      params.session_id === 'warning-resume-a'
        ? [
            { role: 'user', text: 'First prompt', timestamp: 1 },
            { role: 'system', text: 'unrelated status', timestamp: 2 },
            { role: 'assistant', text: 'First reply', timestamp: 3 }
          ]
        : [{ role: 'user', text: 'Other conversation', timestamp: 4 }],
    info: { model: 'test', tools: {}, skills: {} }
  }))

  let lifecycle!: ReturnType<typeof useSessionLifecycle>
  const sys = (text: string) => setHistoryItems(previous => [...previous, { role: 'system', text }])

  function Harness() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() },
      gw: { isCanonical: true, request },
      rpc: request,
      scrollRef: { current: null },
      panel: vi.fn(),
      sys,
      setHistoryItems,
      setLastUserMsg: vi.fn(),
      setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn()
    } as any)

    return <Text>warning resume</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  try {
    await lifecycle.resumeById('warning-resume-a')

    const onEvent = createGatewayEventHandler({
      composer: { setInput: vi.fn() },
      gateway: { gw: { request }, rpc: request },
      session: { STARTUP_RESUME_ID: '', newSession: vi.fn(), resumeById: lifecycle.resumeById },
      system: { sys },
      transcript: { appendMessage: (msg: Msg) => setHistoryItems(rows => [...rows, msg]), setHistoryItems },
      submission: { submitRef: { current: vi.fn() } },
      voice: { setProcessing: vi.fn(), setRecording: vi.fn() }
    } as any)

    const warning = { kind: 'model-switch-warning', text: 'Same provider notice', timestamp: 3.5 }
    onEvent({
      type: 'status.update',
      session_id: 'warning-resume-a',
      replay_epoch: 'epoch',
      seq: 7,
      payload: warning
    } as any)
    expect(history.filter(row => row.text === 'warning: Same provider notice')).toHaveLength(1)
    onEvent({
      type: 'status.update',
      session_id: 'warning-resume-a',
      replay_epoch: 'epoch',
      seq: 7,
      payload: warning
    } as any)
    expect(history.filter(row => row.text === 'warning: Same provider notice')).toHaveLength(1)

    await lifecycle.resumeById('warning-resume-b')
    expect(history.some(row => row.text === 'warning: Same provider notice')).toBe(false)
    await lifecycle.resumeById('warning-resume-a')
    expect(history.map(row => row.text).filter(Boolean)).toEqual([
      'First prompt',
      'unrelated status',
      'First reply',
      'warning: Same provider notice'
    ])

    request.mockImplementation(async (_method: string, params: { session_id: string }) => ({
      session_id: params.session_id,
      stored_session_id: params.session_id,
      messages: [
        { role: 'user', text: 'First prompt', timestamp: 1 },
        { role: 'system', text: 'unrelated status', timestamp: 2 },
        { role: 'assistant', text: 'First reply', timestamp: 3 },
        { role: 'system', text: 'warning: Same provider notice', timestamp: 3.5 }
      ],
      info: { model: 'test', tools: {}, skills: {} }
    }))
    await lifecycle.resumeById('warning-resume-a')
    expect(history.filter(row => row.text === 'warning: Same provider notice')).toHaveLength(1)
    expect(history.filter(row => row.text === 'unrelated status')).toHaveLength(1)

    onEvent({
      type: 'status.update',
      session_id: 'warning-resume-a',
      replay_epoch: 'epoch',
      seq: 8,
      payload: { ...warning, timestamp: 5 }
    } as any)
    await lifecycle.resumeById('warning-resume-a')
    expect(history.filter(row => row.text === 'warning: Same provider notice')).toHaveLength(2)
    expect(history.filter(row => row.text === 'unrelated status')).toHaveLength(1)
  } finally {
    instance.unmount()
    resetUiState()
  }
})
