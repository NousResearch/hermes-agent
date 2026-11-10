import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $backtrackPrime, $pendingRewind, armPendingRewind } from '../app/backtrackStore.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useInputHandlers } from '../app/useInputHandlers.js'

class FakeInput extends EventEmitter {
  chunks: string[] = []
  isRaw = false
  isTTY = true
  readableLength = 0
  ref = vi.fn()
  unref = vi.fn()
  setEncoding = vi.fn()
  setRawMode = vi.fn()

  read() {
    const next = this.chunks.shift() ?? null
    this.readableLength = this.chunks.length

    return next
  }

  send(data: string) {
    this.chunks.push(data)
    this.readableLength = this.chunks.length
    this.emit('readable')
  }
}

const wait = () => new Promise(resolve => setTimeout(resolve, 70))
let unmount = () => {}

beforeEach(() => {
  resetUiState()
  resetOverlayState()
  patchUiState({ busy: false, sid: 'owner' })
  $backtrackPrime.set(null)
  $pendingRewind.set(null)
})

afterEach(() => {
  unmount()
  $backtrackPrime.set(null)
  $pendingRewind.set(null)
  resetUiState()
  resetOverlayState()
})

/** The real input handler with one retained queue item and the armed edit's text erased. */
function mount() {
  const setQueueEdit = vi.fn()
  const setInput = vi.fn()
  const sys = vi.fn()
  const stdin = new FakeInput()
  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })

  function Harness() {
    useInputHandlers({
      actions: { countPrompts: () => 2, openBacktrack: vi.fn(), sys },
      composer: {
        actions: { queueDraft: () => 'queued follow-up', setHistoryIdx: vi.fn(), setInput, setQueueEdit },
        refs: { historyRef: { current: [] }, queueRef: { current: [{ text: 'queued follow-up' }] } },
        state: { completions: [], input: '', inputBuf: [], queueEditIdx: null, queuedDisplay: ['queued follow-up'], tokens: [] }
      },
      gateway: { gw: { isCanonical: true } },
      terminal: { hasSelection: false, stdout },
      voice: { recordKey: 'ctrl+b' },
      wheelStep: 3
    } as any)

    return null
  }

  const instance = renderSync(<Harness />, { exitOnCtrlC: false, patchConsole: false, stdin: stdin as any, stdout: stdout as any })

  unmount = () => instance.unmount()

  return { setInput, setQueueEdit, stdin, sys }
}

it('↑ into a queue edit disarms the pending backtrack edit, so Enter resends the queue item and cuts nothing', async () => {
  const send = vi.fn()

  armPendingRewind({ send, sid: 'owner', target: { rowId: 11, text: 'first', turn: 1 } })
  const { setInput, setQueueEdit, stdin, sys } = mount()

  await wait()
  stdin.send('\x1b[A')
  await wait()
  await wait()

  expect(setQueueEdit).toHaveBeenCalledWith(0)
  expect(setInput).toHaveBeenCalledWith('queued follow-up')
  expect($pendingRewind.get()).toBeNull()
  // The user sees that the edit was dropped (nothing was cut).
  expect(sys).toHaveBeenCalledTimes(1)
  expect(send).not.toHaveBeenCalled()
})
