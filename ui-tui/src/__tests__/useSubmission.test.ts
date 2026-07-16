import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComposerActions, ComposerRefs, ComposerState } from '../app/interfaces.js'
import { resetUiState } from '../app/uiStore.js'
import { useSubmission, type UseSubmissionOptions } from '../app/useSubmission.js'
import type { GatewayClient } from '../gatewayClient.js'

const makeStreams = () => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', () => {})

  return { stderr, stdin, stdout }
}

const makeComposerActions = (): ComposerActions => ({
  clearIn: vi.fn(),
  dequeue: vi.fn(),
  enqueue: vi.fn(),
  handleTextPaste: vi.fn(),
  openEditor: vi.fn(),
  pushHistory: vi.fn(),
  removeQueue: vi.fn(),
  replaceQueue: vi.fn(),
  setCompIdx: vi.fn(),
  setHistoryIdx: vi.fn(),
  setInput: vi.fn(),
  setInputBuf: vi.fn(),
  setPasteSnips: vi.fn(),
  setQueueEdit: vi.fn(),
  syncQueue: vi.fn()
})

const makeComposerState = (input: string, text: string, compReplace: number): ComposerState => ({
  compIdx: 0,
  compReplace,
  completions: [{ display: text, text }],
  historyIdx: null,
  input,
  inputBuf: [],
  pasteSnips: [],
  queueEditIdx: null,
  queuedDisplay: []
})

function Harness({ options }: { options: UseSubmissionOptions }) {
  useSubmission(options)

  return null
}

const renderSubmission = (composerState: ComposerState) => {
  const submitRef = { current: vi.fn<(value: string) => void>() }
  const composerActions = makeComposerActions()

  const composerRefs: ComposerRefs = {
    historyDraftRef: { current: '' },
    historyRef: { current: [] },
    queueEditRef: { current: null },
    queueRef: { current: [] },
    submitRef
  }

  const appendMessage = vi.fn()
  const slash = vi.fn<(command: string) => boolean>(() => true)
  const request = vi.fn(() => Promise.resolve(null))

  const options: UseSubmissionOptions = {
    appendMessage,
    composerActions,
    composerRefs,
    composerState,
    gw: { request } as unknown as GatewayClient,
    setLastUserMsg: vi.fn(),
    slashRef: { current: slash },
    submitRef,
    sys: vi.fn()
  }

  const streams = makeStreams()

  const instance = renderSync(React.createElement(Harness, { options }), {
    patchConsole: false,
    stderr: streams.stderr as NodeJS.WriteStream,
    stdin: streams.stdin as NodeJS.ReadStream,
    stdout: streams.stdout as NodeJS.WriteStream
  })

  return { appendMessage, composerActions, instance, request, slash, submitRef }
}

describe('useSubmission completion routing', () => {
  beforeEach(() => resetUiState())

  it('accepts a path completion into the composer without dispatching', () => {
    const harness = renderSubmission(makeComposerState('cd ~/Proj', '~/Projects/hermes-agent', 3))

    try {
      harness.submitRef.current('cd ~/Proj')

      expect(harness.composerActions.setInput).toHaveBeenCalledOnce()
      expect(harness.composerActions.setInput).toHaveBeenCalledWith('cd ~/Projects/hermes-agent')
      expect(harness.composerActions.pushHistory).not.toHaveBeenCalled()
      expect(harness.composerActions.enqueue).not.toHaveBeenCalled()
      expect(harness.composerActions.clearIn).not.toHaveBeenCalled()
      expect(harness.appendMessage).not.toHaveBeenCalled()
      expect(harness.slash).not.toHaveBeenCalled()
      expect(harness.request).not.toHaveBeenCalled()
    } finally {
      harness.instance.unmount()
      harness.instance.cleanup()
    }
  })

  it('dispatches a completed slash command on the same Enter press', () => {
    const harness = renderSubmission(makeComposerState('/ex', 'exit', 1))

    try {
      harness.submitRef.current('/ex')

      expect(harness.slash).toHaveBeenCalledOnce()
      expect(harness.slash).toHaveBeenCalledWith('/exit')
      expect(harness.composerActions.setInput).not.toHaveBeenCalled()
    } finally {
      harness.instance.unmount()
      harness.instance.cleanup()
    }
  })
})
