import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import chalk from 'chalk'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $backtrackPrime, $pendingRewind, cancelPendingRewind, sendPendingRewind } from '../app/backtrackStore.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import type { RewindOverlay } from '../app/interfaces.js'
import { $overlayState, resetOverlayState } from '../app/overlayStore.js'
import { $rewindFocus, $rewindScrollRestore } from '../app/rewindFocusStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { handleEscBacktrack } from '../app/useInputHandlers.js'
import { RewindBrowse } from '../components/rewindBrowse.js'
import { useBacktrackHint } from '../components/rewindFocus.js'
import { DEFAULT_THEME } from '../theme.js'

const flush = () => new Promise(resolve => setImmediate(resolve))
const settle = (ms = 30) => new Promise(resolve => setTimeout(resolve, ms))

const rows = [
  { role: 'user', content: 'ALPHA please', text: 'ALPHA please', row_id: 11 },
  { role: 'assistant', content: 'ALPHA', text: 'ALPHA', row_id: 12 },
  { role: 'user', content: 'BETA please', text: 'BETA please', row_id: 13 },
  { role: 'assistant', content: 'BETA', text: 'BETA', row_id: 14 },
  { role: 'user', content: 'GAMMA please', text: 'GAMMA please', row_id: 15 },
  { role: 'assistant', content: 'GAMMA', text: 'GAMMA', row_id: 16 }
]

function gateway(messages: unknown[] = rows) {
  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.resume') {
      return { session_id: 'owner', revision: 7, execution_generation: 3, messages, info: { model: 'm' } }
    }

    if (method === 'session.mutate' && params.operation === 'rewind') {
      return { session_id: 'owner', operation: 'rewind', revision: 8, execution_generation: 4, rewound_count: 4 }
    }

    throw new Error(`unknown_method: ${method}`)
  })

  return { isCanonical: true, request }
}

function escHarness(messages: unknown[] = rows) {
  const gw = gateway(messages)
  const prompts = messages.filter((row: any) => row.role === 'user').length
  const actions = { appendMessage: vi.fn(), countPrompts: () => prompts, openBacktrack: vi.fn(), sys: vi.fn() } as any
  const cActions = { clearIn: vi.fn(), pushHistory: vi.fn() } as any
  const cState = { input: '', inputBuf: [] } as any

  const esc = (over: Record<string, unknown> = {}) =>
    handleEscBacktrack({
      actions,
      cActions,
      cState,
      gw: gw as any,
      hasSelection: false,
      isBlocked: false,
      isDouble: false,
      isEsc: true,
      isVoiceKey: false,
      live: { busy: false, sid: 'owner' } as any,
      ...over
    })

  return { actions, cActions, cState, esc, gw }
}

const colorLevel = chalk.level

afterEach(() => {
  chalk.level = colorLevel
})

beforeEach(() => {
  chalk.level = 3
  resetUiState()
  resetOverlayState()
  $backtrackPrime.set(null)
  $pendingRewind.set(null)
  $rewindFocus.set(null)
  $rewindScrollRestore.set(false)
  patchUiState({
    sid: 'owner',
    info: { model: 'm', skills: {}, tools: {}, execution_epoch: '1', execution_generation: 3 } as any
  })
})

describe('Esc backtrack prime', () => {
  it('first Esc primes from the transcript on screen (no RPC); second Esc opens browsing, with no time window', async () => {
    const { actions, esc, gw } = escHarness()

    expect(esc()).toBe(true)
    expect($backtrackPrime.get()).toEqual({ count: 3, sid: 'owner' })
    await flush()
    expect(gw.request).not.toHaveBeenCalled()
    expect(actions.openBacktrack).not.toHaveBeenCalled()

    await settle(600) // longer than the old 500 ms double-Esc window
    expect(esc()).toBe(true)
    expect(actions.openBacktrack).toHaveBeenCalledTimes(1)
    expect($backtrackPrime.get()).toBeNull()
  })

  it('any other key cancels the prime and still acts normally', async () => {
    const { actions, esc } = escHarness()

    esc()
    await flush()
    expect(esc({ isEsc: false })).toBe(false)
    expect($backtrackPrime.get()).toBeNull()
    esc()
    expect(actions.openBacktrack).not.toHaveBeenCalled()
  })

  it('zero earlier prompts: primes silently, second Esc says there is nothing to edit', async () => {
    const { actions, esc } = escHarness([])
    const { request, slash, sys } = slashHarness([])

    actions.openBacktrack.mockImplementation(() => slash('/rewind'))
    esc()
    expect($backtrackPrime.get()?.count).toBe(0)
    expect(hintFrame()).toBeNull()
    esc()
    await flush()
    expect(request).toHaveBeenCalledWith('session.resume', { session_id: 'owner' })
    expect(sys).toHaveBeenCalledWith('No previous message to edit.')
    expect($overlayState.get().rewind).toBeFalsy()
  })

  it('a draft, a running turn or an overlay is not a backtrack Esc (main behaviour kept)', () => {
    const { actions, cActions, cState, esc } = escHarness()

    expect(esc({ live: { busy: true, sid: 'owner' } })).toBe(false)
    expect(esc({ isBlocked: true })).toBe(false)
    cState.input = 'draft'
    expect(esc()).toBe(false)
    expect(esc({ isDouble: true })).toBe(true)
    expect(cActions.pushHistory).toHaveBeenCalledWith('draft')
    expect(cActions.clearIn).toHaveBeenCalled()
    expect(actions.openBacktrack).not.toHaveBeenCalled()
    expect($backtrackPrime.get()).toBeNull()
  })
})

function hintFrame() {
  let hint: null | string = 'unset'

  function Probe() {
    hint = useBacktrackHint()

    return null
  }

  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  const instance = renderSync(React.createElement(Probe), { patchConsole: false, stdout: stdout as any })
  instance.unmount()

  return hint
}

describe('composer hint', () => {
  it('shows the Codex prime hint only with an earlier prompt, and the pending-edit hint when armed', () => {
    $backtrackPrime.set({ count: 0, sid: 'owner' })
    expect(hintFrame()).toBeNull()
    $backtrackPrime.set({ count: 2, sid: 'owner' })
    expect(hintFrame()).toBe('esc again to edit previous message')
    $backtrackPrime.set(null)
    $pendingRewind.set({ send: vi.fn(), sid: 'owner', target: { rowId: 13, text: 'BETA please', turn: 2 } })
    expect(hintFrame()).toBe('↺ Editing prompt 2 — Enter rewinds here and sends · Esc Esc cancels')
  })
})

function slashHarness(messages: unknown[] = rows) {
  const gw = gateway(messages)
  const composer = { enqueue: vi.fn(), setInput: vi.fn() }
  const send = vi.fn()
  const sys = vi.fn()

  const ctx = {
    slashFlightRef: { current: 0 },
    gateway: { gw, rpc: gw.request },
    transcript: {
      page: vi.fn(),
      panel: vi.fn(),
      send,
      sys,
      setHistoryItems: vi.fn(),
      trimLastExchange: (i: unknown[]) => i
    },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { newSession: vi.fn(), resumeById: vi.fn() },
    composer,
    voice: {}
  }

  return { composer, request: gw.request, send, slash: createSlashHandler(ctx as any), sys }
}

const edit = (text: string) => ({ display: text, text })

const mutations = (request: ReturnType<typeof gateway>['request']) =>
  request.mock.calls.filter(([method]) => method === 'session.mutate')

describe('browse → Enter → send', () => {
  it('Enter prefills the composer and cuts nothing; sending rewinds to the picked row, then sends', async () => {
    const { composer, request, send, slash } = slashHarness()

    slash('/rewind')
    await flush()
    const browse = $overlayState.get().rewind!

    expect(browse.targets.map(target => target.rowId)).toEqual([11, 13, 15])
    expect(browse.messages?.map(msg => msg.text)).toContain('GAMMA please')

    browse.onEdit(browse.targets[1]!)
    expect(composer.setInput).toHaveBeenCalledWith('BETA please')
    await flush()
    expect(mutations(request)).toHaveLength(0)

    expect(sendPendingRewind(edit('BETA please v2'), 'owner')).toBe(true)
    await flush()
    await flush()
    await flush()

    expect(mutations(request)).toHaveLength(1)
    expect(mutations(request)[0]![1]).toMatchObject({ operation: 'rewind', payload: { target_message_id: 13 } })
    expect(send).toHaveBeenCalledWith('BETA please v2')
    expect($pendingRewind.get()).toBeNull()
  })

  it('a slash command, another session or an explicit cancel disarms the edit and cuts nothing', async () => {
    const { request, slash, sys } = slashHarness()

    slash('/rewind')
    await flush()
    const browse = $overlayState.get().rewind!

    browse.onEdit(browse.targets[0]!)
    expect(sendPendingRewind(edit('/help'), 'owner')).toBe(false)
    expect($pendingRewind.get()).toBeNull()

    browse.onEdit(browse.targets[0]!)
    expect(sendPendingRewind(edit('ALPHA v2'), 'other')).toBe(false)

    browse.onEdit(browse.targets[0]!)
    expect(cancelPendingRewind(sys)).toBe(true)
    expect(sys).toHaveBeenCalledWith('↺ Rewind cancelled; nothing was removed.')
    await flush()
    expect(mutations(request)).toHaveLength(0)
  })
})

class FakeInput extends EventEmitter {
  chunks: string[] = []
  isRaw = false
  isTTY = true
  readableLength = 0
  ref = vi.fn()
  setEncoding = vi.fn()
  unref = vi.fn()

  read() {
    const next = this.chunks.shift() ?? null
    this.readableLength = this.chunks.length

    return next
  }

  send(chunk: string) {
    this.chunks.push(chunk)
    this.readableLength = this.chunks.length
    this.emit('readable')
  }

  setRawMode = vi.fn((enabled: boolean) => {
    this.isRaw = enabled
  })
}

const targets = [
  { rowId: 11, text: 'ALPHA please', turn: 1 },
  { rowId: 13, text: 'BETA please', throughRowId: 12, turn: 2 },
  { rowId: 15, text: 'GAMMA please', throughRowId: 14, turn: 3 }
]

function mountBrowse(native = false) {
  const handlers = { onBack: vi.fn(), onClose: vi.fn(), onEdit: vi.fn(), onFork: vi.fn(), onType: vi.fn() }

  const req: RewindOverlay = {
    messages: rows.map(row => ({ role: row.role as 'assistant' | 'user', text: row.text })),
    onSubmit: vi.fn(),
    targets,
    ...handlers
  }

  const stdin = new FakeInput()
  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  const frames: string[] = []
  stdout.on('data', chunk => frames.push(chunk.toString()))

  const instance = renderSync(<RewindBrowse cols={80} native={native} {...handlers} req={req} t={DEFAULT_THEME} />, {
    patchConsole: false,
    stdin: stdin as any,
    stdout: stdout as any
  })

  const key = async (seq: string) => {
    stdin.send(seq)
    await settle()
  }

  return { frames, handlers, instance, key }
}

describe('RewindBrowse keys', () => {
  it('starts on the latest prompt; ←/h older and →/l newer clamp; Enter edits the selected one', async () => {
    const { handlers, instance, key } = mountBrowse()

    await settle()
    expect($rewindFocus.get()?.turn).toBe(3)
    await key('\x1b[C')
    expect($rewindFocus.get()?.turn).toBe(3)
    await key('\x1b[D')
    await key('h')
    await key('h')
    expect($rewindFocus.get()?.turn).toBe(1)
    await key('l')
    expect($rewindFocus.get()?.turn).toBe(2)
    await key('\r')
    expect(handlers.onEdit).toHaveBeenCalledWith(targets[1])
    instance.unmount()
    expect($rewindFocus.get()).toBeNull()
  })

  it('Esc goes back with the scroll restored, q closes, f forks, any other key closes and types', async () => {
    const { handlers, instance, key } = mountBrowse()

    await settle()
    await key('x')
    expect(handlers.onType).toHaveBeenCalledWith('x')
    await key('q')
    expect(handlers.onClose).toHaveBeenCalled()
    await key('f')
    expect(handlers.onFork).toHaveBeenCalledWith(targets[2])
    await key('\x1b')
    await settle(80)
    expect(handlers.onBack).toHaveBeenCalled()
    expect($rewindScrollRestore.get()).toBe(true)
    instance.unmount()
  })

  it('native mode draws a transcript pager with the selected prompt in reverse video and the hint', async () => {
    const { frames, instance, key } = mountBrowse(true)

    await settle()
    const first = frames.join('')
    // SGR 7 (inverse) opens right before the selected prompt's text: the latest one first.
    const inverse = (text: string) => new RegExp(`\\x1b\\[(?:[0-9;]*;)?7m[^\\x1b]*› ${text}`)

    expect(stripAnsi(first)).toContain(
      'Browsing transcript · ↑↓/jk scroll · ←→/hl prompts · ↵ edit · f fork · esc back'
    )
    expect(first).toMatch(inverse('GAMMA please'))
    expect(first).not.toMatch(inverse('BETA please'))

    const seen = frames.length
    await key('h')
    const next = frames.slice(seen).join('')

    expect(next).toMatch(inverse('BETA please'))
    expect(next).not.toMatch(inverse('GAMMA please'))
    instance.unmount()
  })
})
