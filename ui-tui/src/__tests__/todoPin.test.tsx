import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, describe, expect, it } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { AppLayoutProps } from '../app/interfaces.js'
import { patchTurnState, resetTurnState } from '../app/turnStore.js'
import { resetUiState } from '../app/uiStore.js'
import { AppLayout } from '../components/appLayout.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_VOICE_RECORD_KEY } from '../lib/platform.js'
import type { Msg, TodoItem } from '../types.js'

// The live Todo panel must stay visible while the transcript scrolls: it is
// pinned between the scroll area and the composer, not a child of the latest
// user-message row inside ScrollBox. These tests paint the real AppLayout and
// scroll the latest user row out of the virtual window — the todo must stay.

const WATCHWORD = 'pinned-todo-watchword-122506'

const todo = (content: string): TodoItem => ({ content, id: `todo-${content}`, status: 'in_progress' })

const userMsg = (text: string): Msg => ({ role: 'user', text })

const assistantMsg = (text: string): Msg => ({ role: 'assistant', text })

const paint = (props: AppLayoutProps, columns = 80, rows = 24): string => {
  const stdout = Object.assign(new PassThrough(), { columns, isTTY: false, rows })
  const frames: string[] = []
  stdout.on('data', chunk => {
    frames.push(chunk.toString())
  })

  const gateway = {
    gw: {} as GatewayClient,
    rpc: async () => null
  }

  const view = renderSync(
    <GatewayProvider value={gateway}>
      <AppLayout {...props} />
    </GatewayProvider>,
    {
    patchConsole: false,
    stderr: new PassThrough() as unknown as NodeJS.WriteStream,
    stdin: new PassThrough() as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  view.unmount()
  view.cleanup()

  return stripAnsi(frames.join(''))
}

const baseProps = (): AppLayoutProps => {
  const noop = () => undefined

  return {
    actions: {
      answerApproval: noop,
      answerClarify: noop,
      answerClarifyQuestion: noop,
      answerSecret: noop,
      answerSudo: noop,
      answerVaultUnlock: noop,
      clearSelection: noop,
      activateLiveSession: noop,
      closeLiveSession: async () => null,
      newLiveSession: noop,
      newPromptSession: noop,
      onModelSelect: noop,
      resumeById: noop,
      setStickyPrompt: noop
    },
    composer: {
      cols: 80,
      compIdx: 0,
      completions: [],
      empty: true,
      handleTextPaste: () => null,
      input: '',
      inputBuf: [],
      pagerPageSize: 10,
      queueEditIdx: null,
      queuedDisplay: [],
      submit: noop,
      updateInput: noop,
      voiceRecordKey: DEFAULT_VOICE_RECORD_KEY
    },
    mouseTracking: 'off',
    progress: { showProgressArea: false },
    status: {
      cwdLabel: '~',
      goodVibesTick: 0,
      lastTurnEndedAt: null,
      sessionStartedAt: null,
      sessionTitle: '',
      showStickyPrompt: false,
      statusColor: '',
      stickyPrompt: '',
      turnStartedAt: null,
      voiceLabel: ''
    },
    transcript: {
      historyItems: [],
      scrollRef: { current: null },
      virtualHistory: {
        bottomSpacer: 0,
        end: 0,
        measureRef: () => () => undefined,
        offsets: [],
        start: 0,
        topSpacer: 0
      },
      virtualRows: []
    }
  }
}

const withTranscript = (
  historyItems: Msg[],
  rows: Array<{ index: number; key: string; msg: Msg }>,
  window: { end: number; start: number }
): AppLayoutProps => {
  const props = baseProps()
  props.transcript = {
    ...props.transcript,
    historyItems,
    virtualHistory: { ...props.transcript.virtualHistory, end: window.end, start: window.start },
    virtualRows: rows
  }

  return props
}

describe('pinned Todo panel (#122506)', () => {
  afterEach(() => {
    resetTurnState()
    resetUiState()
  })

  it('keeps the live todos visible after the latest user row scrolls out of view', () => {
    patchTurnState({ todoCollapsed: false, todos: [todo(WATCHWORD)] })

    const history = [userMsg('do the thing'), ...Array.from({ length: 30 }, (_, i) => assistantMsg(`filler ${i}`))]
    const rows = history.map((msg, i) => ({ index: i, key: `row-${i}`, msg }))
    // Latest user row is index 0; the virtual window has scrolled past it.
    const props = withTranscript(history, rows, { end: rows.length, start: 1 })

    expect(paint(props)).toContain(WATCHWORD)
  })

  it('shows the live todos while the latest user row is still in view', () => {
    patchTurnState({ todoCollapsed: false, todos: [todo(WATCHWORD)] })

    const history = [userMsg('do the thing'), assistantMsg('on it')]
    const rows = history.map((msg, i) => ({ index: i, key: `row-${i}`, msg }))
    const props = withTranscript(history, rows, { end: rows.length, start: 0 })

    expect(paint(props)).toContain(WATCHWORD)
  })

  it('takes no space when there are no todos', () => {
    patchTurnState({ todoCollapsed: false, todos: [] })

    const history = [userMsg('do the thing')]
    const rows = history.map((msg, i) => ({ index: i, key: `row-${i}`, msg }))
    const props = withTranscript(history, rows, { end: rows.length, start: 0 })

    expect(paint(props)).not.toContain(WATCHWORD)
  })
})
