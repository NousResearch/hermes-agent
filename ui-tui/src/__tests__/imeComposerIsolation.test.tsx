import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useMemo } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type {
  AppLayoutActions,
  AppLayoutComposerProps,
  AppLayoutStatusProps,
  VirtualHistoryState
} from '../app/interfaces.js'
import { resetUiState } from '../app/uiStore.js'
import { AppLayout } from '../components/appLayout.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { Msg } from '../types.js'

const counters = vi.hoisted(() => ({ messageLine: 0, textInput: 0 }))

vi.mock('../app/usePet.js', () => ({
  usePet: () => ({ enabled: false, grid: null, kitty: null })
}))

vi.mock('../components/messageLine.js', () => ({
  // Deliberately NOT memoized: a memoized stand-in would hide a TranscriptPane
  // re-render behind its own shallow-equal props and gut the guard below.
  MessageLine: () => {
    counters.messageLine++

    return null
  }
}))

vi.mock('../components/streamingAssistant.js', () => ({
  LiveTodoPanel: () => null,
  StreamingAssistant: () => null
}))

vi.mock('../components/textInput.js', () => {
  console.log('PROBE textInput mock factory invoked')

  return {
    TextInput: () => {
      counters.textInput++

      return null
    }
  }
})

const historyItems: Msg[] = [
  { role: 'user', text: '你好，世界' },
  { role: 'assistant', text: 'ready' }
]

const virtualRows = historyItems.map((msg, index) => ({ index, key: `row:${index}`, msg }))

const virtualHistory: VirtualHistoryState = {
  bottomSpacer: 0,
  end: virtualRows.length,
  measureRef: () => () => {},
  offsets: [],
  start: 0,
  topSpacer: 0
}

const baseComposer: AppLayoutComposerProps = {
  cols: 80,
  compIdx: -1,
  completions: [],
  empty: false,
  handleTextPaste: () => null,
  input: '',
  inputBuf: [],
  pagerPageSize: 10,
  queueEditIdx: null,
  queuedDisplay: [],
  submit: () => {},
  updateInput: () => {},
  voiceRecordKey: null
}

const actions = {
  activateLiveSession: () => {},
  answerApproval: () => {},
  answerClarifyQuestion: () => {},
  answerSecret: () => {},
  answerSudo: () => {},
  answerVaultUnlock: () => {},
  cancelClarify: () => {},
  clearSelection: () => {},
  closeLiveSession: async () => null,
  newLiveSession: () => {},
  newPromptSession: () => {},
  onModelSelect: () => {},
  resumeById: () => {},
  setStickyPrompt: () => {}
} satisfies AppLayoutActions

const status = {
  cwdLabel: '',
  goodVibesTick: 0,
  lastTurnEndedAt: null,
  sessionStartedAt: null,
  sessionTitle: '',
  showStickyPrompt: false,
  statusColor: '',
  stickyPrompt: '',
  turnStartedAt: null,
  voiceLabel: ''
} satisfies AppLayoutStatusProps

const progress = { showProgressArea: false }

const scrollRef = { current: null }

// Module-level constants mirror useMainApp's memoized `appTranscript` /
// `appProgress` objects: their identity must not flip when only the composer
// input changes, or the harness itself defeats TranscriptPane's memo.
const transcript = { historyItems, scrollRef, virtualHistory, virtualRows }


const gatewayValue = {
  gw: { request: async () => ({}) } as unknown as GatewayClient,
  rpc: {}
}

/**
 * Mirrors useMainApp's `appComposer` memo: the composer object identity flips
 * on every keystroke while `cols` stays stable — the exact prop shape every
 * committed character sends down AppLayout.
 */
function Harness({ cols, input }: { cols: number; input: string }) {
  const composer = useMemo(() => ({ ...baseComposer, cols, input }), [cols, input])

  return (
    <GatewayProvider value={gatewayValue as never}>
      <AppLayout
        actions={actions}
        composer={composer}
        mouseTracking={'off'}
        progress={progress}
        status={status}
        transcript={transcript}
      />
    </GatewayProvider>
  )
}

describe('AppLayout keystroke isolation (#84349)', () => {
  afterEach(() => {
    resetUiState()
  })

  it('a committed non-ASCII string re-renders the composer but not the transcript rows', async () => {
    const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })

    const stdin = Object.assign(new PassThrough(), {
      isTTY: true,
      setRawMode: () => {},
      ref: () => {},
      unref: () => {}
    })

    const view = renderSync(<Harness cols={80} input="" />, {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    })

    try {
      // Ink renders through a concurrent root, so wait for the initial mount
      // to commit before taking any baseline.
      await vi.waitFor(() => expect(counters.textInput).toBeGreaterThanOrEqual(1))
      // Both history rows mounted.
      expect(counters.messageLine).toBe(historyItems.length)
      const transcriptBaseline = counters.messageLine
      const composerBaseline = counters.textInput

      // An IME/CJK commit flips the composer object identity (input changed)
      // while every transcript prop stays identical. Before #84349's fix the
      // whole-object `composer` prop defeated TranscriptPane's memo and every
      // committed character re-rendered the entire transcript. Waiting on the
      // composer counter lands us after the same commit that would have
      // re-rendered the transcript, so the equality below is final.
      view.rerender(<Harness cols={80} input="你好，世界" />)
      await vi.waitFor(() => expect(counters.textInput).toBeGreaterThan(composerBaseline))
      expect(counters.messageLine).toBe(transcriptBaseline)

      // Control: a terminal resize (cols) is a real transcript input, so the
      // transcript must re-render — proves the zero above is isolation, not a
      // dead harness.
      view.rerender(<Harness cols={100} input="你好，世界" />)
      await vi.waitFor(() => expect(counters.messageLine).toBeGreaterThan(transcriptBaseline))
    } finally {
      view.unmount()
      view.cleanup()
    }
  })
})
