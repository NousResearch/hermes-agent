import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $backtrackPrime } from '../app/backtrackStore.js'
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
})

afterEach(() => {
  unmount()
  $backtrackPrime.set(null)
  resetUiState()
  resetOverlayState()
})

/** The real input handler on an idle, canonical session with two earlier prompts on screen. */
function mount(queueEditIdx: null | number) {
  const clearIn = vi.fn()
  const openBacktrack = vi.fn()
  const stdin = new FakeInput()
  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })

  function Harness() {
    useInputHandlers({
      actions: { countPrompts: () => 2, openBacktrack, sys: vi.fn() },
      composer: {
        actions: { clearIn },
        refs: {},
        state: { completions: [], input: '', inputBuf: [], queueEditIdx, tokens: [] }
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

  return { clearIn, openBacktrack, stdin }
}

it('one Esc cancels a queue edit whose text was erased, and does not prime backtrack', async () => {
  const { clearIn, stdin } = mount(0)

  await wait()
  stdin.send('\x1b')
  await wait()
  await wait()

  expect(clearIn).toHaveBeenCalledTimes(1)
  expect($backtrackPrime.get()).toBeNull()
})

it('without a queue edit the same Esc still primes backtrack', async () => {
  const { clearIn, stdin } = mount(null)

  await wait()
  stdin.send('\x1b')
  await wait()
  await wait()

  expect(clearIn).not.toHaveBeenCalled()
  expect($backtrackPrime.get()).toEqual({ count: 2, sid: 'owner' })
})
