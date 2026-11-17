import { PassThrough } from 'node:stream'

import { renderSync, useInput } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createPtyStop } from '../../../web/src/lib/pty-stop.js'
import { INITIAL_STATE, parseMultipleKeypresses } from '../../packages/hermes-ink/src/ink/parse-keypress.js'
import { useDashboardStop } from '../app/dashboardStop.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

class TtyStream extends PassThrough {
  isTTY = true
  columns = 80
  rows = 24
  setRawMode() { return this }
  ref() { return this }
  unref() { return this }
}

it('carries the browser control through real fragmented terminal input to the canonical controller', async () => {
  resetUiState()
  turnController.reset()
  patchUiState({ sid: 'profile-session', busy: true, info: {
    model: 'test', skills: {}, tools: {}, profile_name: 'owned-profile',
    execution_epoch: 'owned-epoch', execution_generation: 7
  } })
  const stdin = new TtyStream()
  const stdout = new TtyStream()
  const stderr = new TtyStream()
  const browser = createPtyStop(() => {})
  const sent: string[] = []
  const request = vi.fn(async () => ({ execution_state: 'running', execution_generation: 7 }))
  const gw = { isCanonical: true, request } as unknown as Parameters<typeof useDashboardStop>[0]['gw']
  const appendMessage = vi.fn()
  const sys = vi.fn()
  let outputState = INITIAL_STATE
  let advertised = false

  stdout.on('data', chunk => {
    const [items, next] = parseMultipleKeypresses(outputState, String(chunk))
    outputState = next

    for (const item of items) {
      if (item.kind === 'response' && item.response.type === 'osc' && item.response.code === 6973) {
        advertised = true
        browser.receive(item.response.data)
      }
    }
  })

  function Harness() {
    useInput(() => {})
    useDashboardStop({ enabled: true, stdout: stdout as unknown as NodeJS.WriteStream, gw, appendMessage, sys })

    return null
  }

  const instance = renderSync(React.createElement(Harness), {
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream,
    stderr: stderr as unknown as NodeJS.WriteStream,
    exitOnCtrlC: false
  })

  try {
    await vi.waitFor(() => expect(advertised).toBe(true), { timeout: 2000 })
    browser.open({ readyState: 1, send(frame) {
      sent.push(frame)
      stdin.write(frame.slice(0, 6))
      queueMicrotask(() => stdin.write(frame.slice(6)))
    } })
    await vi.waitFor(() => expect(browser.enabled()).toBe(true), { timeout: 2000 })
    browser.stop()
    await vi.waitFor(() => expect(turnController.interrupted).toBe(true), { timeout: 2000 })
    expect(request).toHaveBeenCalledExactlyOnceWith('session.interrupt', {
      session_id: 'profile-session', execution_generation: 7
    })
    expect(getUiState().busy).toBe(false)
    expect(browser.enabled()).toBe(false)
    expect(sent).toHaveLength(2)
    expect(sent.every(frame => frame.startsWith('\x1b]6973;') && !frame.includes('\x03'))).toBe(true)
  } finally {
    browser.close()
    instance.unmount()
    turnController.clearStatusTimer()
    resetUiState()
  }
})
