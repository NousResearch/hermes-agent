import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useMainApp } from '../app/useMainApp.js'
import { connectionLost } from '../app/userMessages.js'

// B2-20: Ink keeps `sid` across a socket close (gatewayConnected:false). The 1.5 s refresh poll must
// stop while offline (each request re-dials through ensureAttachedWebSocket and collapses the
// reconnect backoff), and every failed redial's `exit` must not append another connectionLost line.
const mounted: Array<() => void> = []
afterEach(() => { for (const unmount of mounted.splice(0)) { unmount() } vi.useRealTimers(); resetUiState() })

it('an offline session stops the refresh poll and reports the lost connection once', async () => {
  vi.useFakeTimers()
  resetUiState()

  const gw = Object.assign(new EventEmitter(), {
    attached: true, isCanonical: true, drain: () => {}, getLogTail: () => '', kill: () => {}, recordLog: () => {},
    request: vi.fn((_method: string) => new Promise(() => {}))
  })

  let app!: ReturnType<typeof useMainApp>

  function Harness() {
    app = useMainApp(gw as never)

    return null
  }

  const stream = () => Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  // useMainApp's input hooks need raw mode, or Ink swaps the tree for an error panel.
  const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: () => {}, setRawMode: () => {}, unref: () => {} })
  const instance = renderSync(React.createElement(Harness), { patchConsole: false, stderr: stream() as never, stdin: stdin as never, stdout: stream() as never })
  mounted.push(() => { instance.unmount(); instance.cleanup() })

  patchUiState({ sid: 'owner', storedSid: 'owner', gatewayConnected: true })
  await vi.advanceTimersByTimeAsync(3100)
  const polls = () => gw.request.mock.calls.filter(([method]) => method === 'subagent.list').length
  expect(polls()).toBeGreaterThan(1)

  for (let attempt = 0; attempt < 3; attempt++) { gw.emit('exit', 1006) }
  await vi.advanceTimersByTimeAsync(0)
  expect(getUiState().gatewayConnected).toBe(false)
  const offline = polls()
  await vi.advanceTimersByTimeAsync(6000)
  expect(polls()).toBe(offline)
  const lost = app.appTranscript.historyItems.filter(item => item.text === connectionLost())
  expect(lost).toHaveLength(1)
})
