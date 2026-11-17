import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { artifactLinkText, turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { MessageLine } from '../components/messageLine.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const VIEWER = 'https://portal.example.com/a/01hx'

const ctx = () =>
  ({
    composer: { dequeue: () => undefined, queueEditRef: { current: null }, sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: { current: 80 },
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

const linkRows = () => getTurnState().streamSegments.filter(m => m.linkUrl)

describe('hosted artifact viewer links', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ sid: 's1', info: { model: 'test', skills: {}, tools: {} } as any })
  })

  const complete = (payload: Record<string, unknown>) => {
    const onEvent = createGatewayEventHandler(ctx())
    onEvent({ type: 'message.start', payload: {}, session_id: 's1' } as any)
    onEvent({ type: 'tool.start', payload: { tool_id: 'c1', name: payload.name, context: '' }, session_id: 's1' } as any)
    onEvent({ type: 'tool.complete', payload: { tool_id: 'c1', ...payload }, session_id: 's1' } as any)
  }

  it('hangs the viewer link under the tool call that returned it', () => {
    complete({
      name: 'mcp__hermes_artifacts__artifact_create',
      artifact_link: { url: VIEWER, title: 'Plan' },
      result: { result: '{"content_url":"https://01hx.frame.example.com/__av?t=secret"}' }
    })

    expect(linkRows()).toEqual([
      { kind: 'event', linkUrl: VIEWER, role: 'system', text: `Artifact \u201cPlan\u201d: ${VIEWER}` }
    ])
    expect(JSON.stringify(getTurnState().streamSegments)).not.toContain('secret')
  })

  it('adds nothing for a call without a link', () => {
    complete({ name: 'mcp__hermes_artifacts__artifact_update', result: { error: 'stale_base' } })
    complete({ name: 'terminal', artifact_link: null })
    expect(linkRows()).toEqual([])
  })

  it('labels an untitled artifact by its link alone', () => {
    expect(artifactLinkText({ url: VIEWER })).toBe(`Artifact: ${VIEWER}`)
  })
})

describe('artifact link row rendering', () => {
  const unmounts: Array<() => void> = []

  afterEach(() => unmounts.splice(0).forEach(fn => fn()))

  const render = (msg: Msg) => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 120, isTTY: false, rows: 20 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(<MessageLine cols={120} msg={msg} t={DEFAULT_THEME} />, {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    unmounts.push(() => {
      instance.unmount()
      instance.cleanup()
    })

    return output
  }

  it('makes the row an OSC 8 hyperlink to the viewer page', () => {
    const output = render({ kind: 'event', linkUrl: VIEWER, role: 'system', text: `Artifact: ${VIEWER}` })

    expect(output).toContain(`\x1b]8;`)
    expect(output).toContain(VIEWER)
  })

  it('leaves ordinary event rows unlinked', () => {
    expect(render({ kind: 'event', role: 'system', text: 'Model switched' })).not.toContain('\x1b]8;')
  })
})
