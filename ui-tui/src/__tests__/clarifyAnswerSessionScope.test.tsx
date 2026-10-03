import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useMainApp } from '../app/useMainApp.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { ClarifyLockResponse } from '../gatewayTypes.js'
import { toolTrailLabel } from '../lib/text.js'

const mounted = new Set<() => void>()

const mountClarify = () => {
  turnController.fullReset()
  resetUiState()
  resetOverlayState()
  patchUiState({ busy: true, sid: 'session-a' })
  patchOverlayState({
    clarify: {
      answers: { first: 'red' },
      questions: [
        { qid: 'first', question: 'First colour?' },
        { qid: 'last', question: 'Last colour?' }
      ],
      requestId: 'request-a'
    }
  })

  let resolveLock!: (value: ClarifyLockResponse | null) => void
  let rejectLock!: (error: Error) => void

  const lock = new Promise<ClarifyLockResponse | null>((resolve, reject) => {
    resolveLock = resolve
    rejectLock = reject
  })

  const gateway = Object.assign(new EventEmitter(), {
    drain: () => {},
    request: vi.fn((method: string) => (method === 'clarify.lock' ? lock : new Promise(() => {})))
  })

  let app!: ReturnType<typeof useMainApp>

  function Probe() {
    app = useMainApp(gateway as unknown as GatewayClient)

    return null
  }

  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: true, ref: () => {}, setRawMode: () => {}, unref: () => {} })

  const instance = renderSync(<Probe />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  const close = () => {
    instance.unmount()
    instance.cleanup()
    mounted.delete(close)
  }

  mounted.add(close)

  return {
    close,
    gateway,
    read: () => app,
    render: () => instance.rerender(<Probe />),
    settle: async (outcome: ClarifyLockResponse | Error | null) => {
      if (outcome instanceof Error) {
        rejectLock(outcome)
      } else {
        resolveLock(outcome)
      }

      // Drain the RPC's promise chain, then synchronously commit the real hook's
      // queued transcript updates. No timing-based negative assertion.
      await new Promise<void>(resolve => setImmediate(resolve))
      instance.rerender(<Probe />)
    }
  }
}

afterEach(() => {
  for (const close of mounted) {
    close()
  }

  turnController.fullReset()
  resetOverlayState()
  resetUiState()
})

describe('clarify answer session ownership', () => {
  it('keeps every late lock outcome out of the replacement session', async () => {
    const outcomes: Array<ClarifyLockResponse | Error | null> = [
      { status: 'expired' },
      null,
      new Error('old session transport failed'),
      { remaining: [], status: 'ok' },
      { remaining: ['last'], status: 'ok' }
    ]

    for (const outcome of outcomes) {
      const probe = mountClarify()
      probe.read().appActions.answerClarifyQuestion('last', 'blue')
      expect(probe.gateway.request).toHaveBeenCalledWith('clarify.lock', {
        answer: 'blue',
        question_id: 'last',
        request_id: 'request-a'
      })

      turnController.fullReset()
      patchUiState({ sid: 'session-b', status: 'waiting for the new answer' })
      const replacement = { questions: [{ qid: 'new', question: 'New session?' }], requestId: 'request-b' }
      patchOverlayState({ clarify: replacement })
      const label = toolTrailLabel('clarify')
      turnController.reservePersistedToolLabel(label, replacement.requestId)
      probe.render()
      const before = probe.read().appTranscript.historyItems

      await probe.settle(outcome)

      expect(probe.read().appTranscript.historyItems).toEqual(before)
      expect(getOverlayState().clarify).toEqual(replacement)
      expect(getUiState().status).toBe('waiting for the new answer')
      expect(turnController.persistedToolLabels.has(label)).toBe(true)
      probe.close()
    }
  })

  it('records a completed answer in its own session without changing a newer prompt', async () => {
    const probe = mountClarify()
    turnController.recordToolStart('tool-a', 'clarify', 'old batch')
    probe.read().appActions.answerClarifyQuestion('last', 'blue')
    probe.gateway.emit('event', {
      payload: { name: 'clarify', tool_id: 'tool-a' },
      session_id: 'session-a',
      type: 'tool.complete'
    })

    const next = { questions: [{ qid: 'next', question: 'Next step?' }], requestId: 'request-next' }
    patchOverlayState({ clarify: next })
    patchUiState({ status: 'waiting for the next answer' })
    const label = toolTrailLabel('clarify')
    turnController.reservePersistedToolLabel(label, next.requestId)
    probe.render()

    await probe.settle({ remaining: [], status: 'ok' })

    expect(probe.read().appTranscript.historyItems.filter(message => message.role === 'user')).toEqual([
      expect.objectContaining({ text: 'First colour? → red\nLast colour? → blue' })
    ])
    expect(probe.read().appTranscript.historyItems.some(message => message.text.includes('timed out'))).toBe(false)
    expect(getOverlayState().clarify).toEqual(next)
    expect(getUiState().status).toBe('waiting for the next answer')
    expect(turnController.persistedToolLabels.has(label)).toBe(true)
  })
})
