import { PassThrough } from 'node:stream'

import { renderSync, Text, useInput } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { InputHandlerContext, OverlayState } from '../app/interfaces.js'
import { $overlayState, getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { useInputHandlers } from '../app/useInputHandlers.js'

// A second useInput stands in for AgentsOverlayPane's own Ctrl+T close
// (agentsOverlay.tsx). It registers BEFORE the app-level handler to pin the
// stricter dispatch order. An input-emitter re-registration can flip
// registration order, and the app-level handler must never undo the
// overlay's close. The overlay is "mounted" only while open (its render-time
// snapshot), like appLayout's pane swap.
function Harness({ ctx }: { ctx: InputHandlerContext }) {
  const overlay = useStore($overlayState)

  useInput((ch, key) => {
    if (overlay.agents && key.ctrl && ch === 't') {
      patchOverlayState({ agents: false, agentsInitialHistoryIndex: 0 })
    }
  })

  useInputHandlers(ctx)

  return React.createElement(Text, null, overlay.agents ? 'AGENTS_OPEN' : 'AGENTS_CLOSED')
}

function renderHarness() {
  const ctx = {
    actions: { die: vi.fn(), interruptTurn: vi.fn(), sys: vi.fn() },
    composer: {
      actions: {
        clearIn: vi.fn(),
        pushHistory: vi.fn(),
        removeQueue: vi.fn(),
        setCompIdx: vi.fn()
      },
      refs: { inputRef: { current: null }, scrollRef: { current: null } },
      state: {
        completions: [],
        historyIdx: null,
        input: '',
        inputBuf: [],
        queueEditIdx: null,
        tokens: []
      }
    },
    gateway: { gw: { publishLocalEvent: vi.fn() } },
    terminal: {
      hasSelection: false,
      scrollRef: { current: null },
      scrollWithSelection: vi.fn(),
      selection: { clearSelection: vi.fn(), copySelection: vi.fn() },
      stdout: { columns: 80, rows: 24 } as unknown as NodeJS.WriteStream
    },
    voice: {
      enabled: false,
      recordKey: { key: 'f9', ctrl: true, meta: false, shift: false, super: false },
      recording: false,
      setProcessing: vi.fn(),
      setRecording: vi.fn(),
      setVoiceEnabled: vi.fn(),
      setVoiceTts: vi.fn()
    },
    wheelStep: 3
  } as unknown as InputHandlerContext

  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })

  const stdin = Object.assign(new PassThrough(), {
    isTTY: true,
    setRawMode: () => {},
    ref: () => {},
    unref: () => {}
  })

  let output = ''
  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })

  const view = renderSync(React.createElement(Harness, { ctx }), {
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stderr: new PassThrough() as unknown as NodeJS.WriteStream,
    patchConsole: false,
    exitOnCtrlC: false
  })

  return { getOutput: () => output, stdin, view }
}

const pendingApproval = { command: 'rm -rf /tmp/x', description: 'cleanup' } as unknown as OverlayState['approval']

describe('Ctrl+T opens the agents overlay while a prompt overlay is pending', () => {
  let view: ReturnType<typeof renderHarness>['view'] | undefined

  afterEach(() => {
    view?.unmount()
    view = undefined
    resetOverlayState()
  })

  const open = async () => {
    resetOverlayState()
    patchOverlayState({ approval: pendingApproval })
    const h = renderHarness()
    view = h.view
    h.stdin.write('\x14')
    await vi.waitFor(() => expect(h.getOutput()).toContain('AGENTS_OPEN'))

    return h
  }

  it('opens the overlay on Ctrl+T and keeps the prompt pending', async () => {
    await open()
    expect(getOverlayState().approval).toBe(pendingApproval)
  })

  it('closes again on the next Ctrl+T, prompt still pending', async () => {
    const { stdin } = await open()
    stdin.write('\x14')
    await vi.waitFor(() => expect(getOverlayState().agents).toBe(false))
    expect(getOverlayState().approval).toBe(pendingApproval)
  })

  it('Ctrl+C in the overlay closes only the overlay, never answers the prompt', async () => {
    const { stdin } = await open()
    stdin.write('\x03')
    await vi.waitFor(() => expect(getOverlayState().agents).toBe(false))
    expect(getOverlayState().approval).toBe(pendingApproval)
  })
})
