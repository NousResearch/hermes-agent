import { PassThrough } from 'stream'

import { renderSync, Text } from '@hermes/ink'
import type * as Ink from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setInputSelection } from '../app/inputSelectionStore.js'
import type { InputHandlerContext } from '../app/interfaces.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useInputHandlers } from '../app/useInputHandlers.js'

let inputHandler: null | ((input: string, key: Record<string, boolean>) => void) = null

vi.mock('@hermes/ink', async importOriginal => {
  const actual = await importOriginal<typeof Ink>()

  return {
    ...actual,
    useInput: (handler: (input: string, key: Record<string, boolean>) => void) => {
      inputHandler = handler
    }
  }
})

const ref = <T,>(current: T) => ({ current })
const fn = () => vi.fn()

const buildContext = (): InputHandlerContext =>
  ({
    actions: {
      answerClarify: fn(),
      appendMessage: fn(),
      die: fn(),
      dispatchSubmission: fn(),
      guardBusySessionSwitch: vi.fn(() => false),
      newSession: fn(),
      sys: fn()
    },
    composer: {
      actions: {
        attachClipboardImage: fn(),
        attachImagePath: fn(),
        clearIn: fn(),
        dequeue: vi.fn(() => undefined),
        enqueue: fn(),
        handleTextPaste: vi.fn(async () => null),
        openEditor: vi.fn(async () => undefined),
        prependQueue: fn(),
        pushHistory: fn(),
        removeQueue: fn(),
        setCompIdx: fn(),
        setComposerTokens: fn(),
        setHistoryIdx: fn(),
        setInput: fn(),
        setInputBuf: fn(),
        setQueueEdit: fn(),
        syncTokens: fn(),
        takeQueue: vi.fn(() => undefined)
      },
      refs: {
        historyDraftRef: ref(''),
        historyRef: ref<string[]>([]),
        queueEditRef: ref<null | number>(null),
        queueRef: ref([]),
        submitRef: ref(fn()),
        tokensRef: ref([])
      },
      state: {
        compIdx: 0,
        compReplace: 0,
        completions: [],
        historyIdx: null,
        input: '',
        inputBuf: [],
        queueEditIdx: null,
        queuedDisplay: [],
        tokens: []
      }
    },
    gateway: {
      gw: {
        publishLocalEvent: fn(),
        request: vi.fn(async () => ({}))
      },
      rpc: vi.fn(async () => null)
    },
    terminal: {
      hasSelection: false,
      scrollRef: ref(null),
      scrollWithSelection: fn(),
      selection: {
        captureScrolledRows: fn(),
        clearSelection: fn(),
        copySelection: vi.fn(async () => ''),
        copySelectionNoClear: vi.fn(async () => ''),
        getState: vi.fn(() => null),
        shiftAnchor: fn(),
        shiftSelection: fn(),
        version: vi.fn(() => 0)
      }
    },
    voice: {
      enabled: false,
      recordKey: { ch: 'b', mod: 'ctrl', raw: 'ctrl+b' },
      recording: false,
      setProcessing: fn(),
      setRecording: fn(),
      setVoiceEnabled: fn(),
      setVoiceTts: fn()
    },
    wheelStep: 3
  }) as unknown as InputHandlerContext

const key = (overrides: Record<string, boolean>) => ({
  ctrl: false,
  downArrow: false,
  escape: false,
  meta: false,
  pageDown: false,
  pageUp: false,
  return: false,
  shift: false,
  super: false,
  tab: false,
  upArrow: false,
  wheelDown: false,
  wheelUp: false,
  ...overrides
})

const mountHandler = (ctx: InputHandlerContext) => {
  function Harness() {
    useInputHandlers(ctx)

    return <Text>input handler harness</Text>
  }

  const stdin = new PassThrough()
  const stdout = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdin, { isTTY: false })
  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stderr, { isTTY: false })

  return renderSync(<Harness />, {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })
}

describe('useInputHandlers busy Ctrl+C routing', () => {
  beforeEach(() => {
    inputHandler = null
    resetOverlayState()
    resetUiState()
    patchUiState({ busy: true, sid: 'busy-session' })
  })

  afterEach(() => {
    setInputSelection(null)
    vi.restoreAllMocks()
  })

  it('calls interruptTurn for raw Ctrl+C during a busy turn', () => {
    const ctx = buildContext()
    const interrupt = vi.spyOn(turnController, 'interruptTurn').mockImplementation(() => undefined)
    const instance = mountHandler(ctx)

    try {
      expect(inputHandler).not.toBeNull()
      inputHandler!('c', key({ ctrl: true }))

      expect(interrupt).toHaveBeenCalledOnce()
      expect(interrupt).toHaveBeenCalledWith({
        appendMessage: ctx.actions.appendMessage,
        gw: ctx.gateway.gw,
        sid: 'busy-session',
        sys: ctx.actions.sys
      })
    } finally {
      instance.unmount()
    }
  })

  it.each([
    ['Ctrl+Super+C', { ctrl: true, super: true }],
    ['Ctrl+Meta+C', { ctrl: true, meta: true }]
  ])('copies the active composer selection for %s without interrupting', (_name, modifiers) => {
    const ctx = buildContext()
    const copy = fn()
    const interrupt = vi.spyOn(turnController, 'interruptTurn').mockImplementation(() => undefined)

    setInputSelection({
      clear: fn(),
      collapseToEnd: fn(),
      copy,
      cut: fn(),
      end: 4,
      start: 1,
      value: 'copy me'
    })

    const instance = mountHandler(ctx)

    try {
      expect(inputHandler).not.toBeNull()
      inputHandler!('c', key(modifiers))

      expect(copy).toHaveBeenCalledOnce()
      expect(interrupt).not.toHaveBeenCalled()
    } finally {
      instance.unmount()
    }
  })
})
