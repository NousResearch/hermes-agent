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
import { $rewindFocus, $rewindScrollRestore, rewindFocusIndex } from '../app/rewindFocusStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { handleEscBacktrack } from '../app/useInputHandlers.js'
import { RewindBrowse } from '../components/rewindBrowse.js'
import { useBacktrackHint } from '../components/rewindFocus.js'
import { pasteTokenLabel } from '../lib/text.js'
import { fitBrowseHint } from '../lib/transcriptPager.js'
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
  const cState = { input: '', inputBuf: [], queueEditIdx: null } as any

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
  const composer = {
    appendPaste: vi.fn(async () => {}),
    appendText: vi.fn(),
    attachImagePath: vi.fn(),
    attachImagePaths: vi.fn(async () => {}),
    enqueue: vi.fn(),
    setInput: vi.fn()
  }
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

function mountBrowse(native = false, cols = 100) {
  const handlers = {
    onBack: vi.fn(),
    onClose: vi.fn(),
    onEdit: vi.fn(),
    onFork: vi.fn(),
    onPaste: vi.fn(),
    onType: vi.fn()
  }

  const req: RewindOverlay = {
    messages: rows.map(row => ({ role: row.role as 'assistant' | 'user', text: row.text })),
    onSubmit: vi.fn(),
    targets,
    ...handlers
  }

  const stdin = new FakeInput()
  const stdout = Object.assign(new PassThrough(), { columns: cols, isTTY: false, rows: 24 })
  const frames: string[] = []
  stdout.on('data', chunk => frames.push(chunk.toString()))

  const instance = renderSync(<RewindBrowse cols={cols} native={native} {...handlers} req={req} t={DEFAULT_THEME} />, {
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

  it('a pasted "f" or "q" is text: it is pasted into the composer, never forks or closes', async () => {
    const { handlers, instance, key } = mountBrowse()

    await settle()
    await key('\x1b[200~f\x1b[201~')
    await key('\x1b[200~q\x1b[201~')
    expect(handlers.onFork).not.toHaveBeenCalled()
    expect(handlers.onClose).not.toHaveBeenCalled()
    await key('\x1b[200~one\r\ntwo\x1b[201~')
    expect(handlers.onPaste.mock.calls.map(([text]) => text)).toEqual(['f', 'q', 'one\ntwo'])
    expect(handlers.onType).not.toHaveBeenCalled()
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

  it('↑/↓ and j/k scroll the native pager; the selection stays on its prompt', async () => {
    const many = Array.from({ length: 12 }, (_, i) => [
      { role: 'user' as const, text: `PROMPT ${i + 1}` },
      { role: 'assistant' as const, text: `reply ${i + 1}` }
    ]).flat()

    const handlers = {
      onBack: vi.fn(),
      onClose: vi.fn(),
      onEdit: vi.fn(),
      onFork: vi.fn(),
      onPaste: vi.fn(),
      onType: vi.fn()
    }
    const manyTargets = Array.from({ length: 12 }, (_, i) => ({
      rowId: 100 + 2 * i,
      text: `PROMPT ${i + 1}`,
      turn: i + 1
    }))
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 24 })
    let screen = ''
    stdout.on('data', chunk => (screen += chunk.toString()))

    const instance = renderSync(
      <RewindBrowse
        cols={100}
        native
        {...handlers}
        req={{ messages: many, onSubmit: vi.fn(), targets: manyTargets, ...handlers }}
        t={DEFAULT_THEME}
      />,
      { patchConsole: false, stdin: stdin as any, stdout: stdout as any }
    )

    await settle()
    const oldest = /› PROMPT 1(?!\d)/

    expect(stripAnsi(screen)).toMatch(/› PROMPT 12/)
    expect(stripAnsi(screen)).not.toMatch(oldest)

    for (let i = 0; i < 40; i++) {
      stdin.send(i % 2 ? 'k' : '\x1b[A')
    }

    await settle(80)
    expect(stripAnsi(screen)).toMatch(oldest)
    expect($rewindFocus.get()?.turn).toBe(12)
    instance.unmount()
  })
})

describe('browse footer at narrow widths (Codex drops hints, never cuts one)', () => {
  const full = 'Browsing transcript · ↑↓/jk scroll · ←→/hl prompts · ↵ edit · f fork · esc back'

  it('sheds whole hints in order as the width shrinks', () => {
    expect(fitBrowseHint(full, 100)).toBe(full)
    expect(fitBrowseHint(full, 78)).toBe('Browsing transcript · ↑↓/jk ←→/hl · ↵ edit · f fork · esc back')
    expect(fitBrowseHint(full, 55)).toBe('Browsing transcript · ↑↓/jk ←→/hl · ↵ edit · esc back')
    expect(fitBrowseHint(full, 40)).toBe('Browsing transcript · ↵ edit · esc back')
    expect(fitBrowseHint(full, 30)).toBe('Browsing transcript · esc back')
    expect(fitBrowseHint(full, 10)).toBe('Browsing transcript')
  })

  it('the native pager at 80 columns draws a whole footer, no ellipsis', async () => {
    const { frames, instance } = mountBrowse(true, 80)

    await settle()
    const screen = stripAnsi(frames.join(''))

    expect(screen).toContain('Browsing transcript · ↑↓/jk ←→/hl · ↵ edit · f fork · esc back')
    expect(screen).not.toContain('…')
    instance.unmount()
  })
})

describe('leaving an armed edit', () => {
  const arm = () =>
    $pendingRewind.set({ send: vi.fn(), sid: 'owner', target: { rowId: 13, text: 'BETA please', turn: 2 } })

  it('Esc Esc on an emptied composer cancels the edit instead of browsing; the next message sends plainly', () => {
    const { actions, esc } = escHarness()

    arm()
    const { send } = $pendingRewind.get()!

    expect(esc()).toBe(true)
    expect(hintFrame()).toBe('↺ Editing prompt 2 — Enter rewinds here and sends · Esc Esc cancels')
    expect(esc()).toBe(true)
    expect(actions.openBacktrack).not.toHaveBeenCalled()
    expect($pendingRewind.get()).toBeNull()
    expect(actions.sys).toHaveBeenCalledWith('↺ Rewind cancelled; nothing was removed.')
    expect(sendPendingRewind(edit('an ordinary message'), 'owner')).toBe(false)
    expect(send).not.toHaveBeenCalled()
  })

  it('opening browse again drops the earlier armed edit', async () => {
    const { request, slash } = slashHarness()

    arm()
    slash('/rewind')
    expect($pendingRewind.get()).toBeNull()
    await flush()
    expect($overlayState.get().rewind).toBeTruthy()
    expect(mutations(request)).toHaveLength(0)
  })
})

describe('default browse finds paste and image prompts', () => {
  const body = Array.from({ length: 7 }, (_, i) => `pasted line ${i + 1}`).join('\n')
  const label = pasteTokenLabel(body, 7)

  const vision =
    "[The user sent an image~ Here's what I can see:\nA blue square on red.]\n" +
    '[If you need a closer look, use vision_analyze with image_url: /cache/native-inputs/a.png ~]\n\n' +
    'What two colors?'

  const transcript = [
    { role: 'user', text: 'ALPHA please' },
    { role: 'assistant', text: 'ALPHA' },
    { pastes: [{ label, text: body }], role: 'user', text: `see ${label} now` },
    { role: 'assistant', text: 'ok' },
    { role: 'user', text: '[[ Image 1 ]] What two colors?' },
    { role: 'assistant', text: 'Blue and red.' }
  ]

  it('matches a paste-chip row and an image row painted at send time (no row id)', () => {
    expect(rewindFocusIndex(transcript, { text: `see ${body} now`, turn: 2 })).toBe(2)
    expect(rewindFocusIndex(transcript, { text: vision, turn: 3 })).toBe(4)
  })

  it('a resumed row is found by its row id, whatever its text', () => {
    const resumed = [
      { role: 'user', rowId: 11, text: 'shown differently' },
      { role: 'user', rowId: 15, text: 'x' }
    ]

    expect(rewindFocusIndex(resumed, { rowId: 15, text: 'stored text', turn: 1 })).toBe(1)
  })
})

describe('Enter on an image prompt', () => {
  it('prefills what the user typed and attaches the image again', async () => {
    const vision =
      "[The user sent an image~ Here's what I can see:\nA blue square on red.]\n" +
      '[If you need a closer look, use vision_analyze with image_url: /cache/native-inputs/a.png ~]\n\n' +
      'What two colors?'

    const stored = [
      { role: 'user', content: 'ALPHA please', text: 'ALPHA please', row_id: 11 },
      { role: 'assistant', content: 'ALPHA', text: 'ALPHA', row_id: 12 },
      { role: 'user', content: vision, text: vision, row_id: 13 },
      { role: 'assistant', content: 'Blue and red.', text: 'Blue and red.', row_id: 14 }
    ]

    const { composer, slash } = slashHarness(stored)

    slash('/rewind')
    await flush()
    const browse = $overlayState.get().rewind!

    browse.onEdit(browse.targets[1]!)
    expect(composer.setInput).toHaveBeenCalledWith('What two colors?')
    expect(composer.attachImagePaths).toHaveBeenCalledWith(['/cache/native-inputs/a.png'], { followTyping: true })
  })

  it('on a remote gateway whose files this machine cannot read, the notes stay in the text and it says so', async () => {
    vi.stubEnv('HERMES_TUI_GATEWAY_URL', 'ws://gateway.example:9119')

    const vision =
      "[The user sent an image~ Here's what I can see:\nA blue square on red.]\n" +
      '[If you need a closer look, use vision_analyze with image_url: /nonexistent-owner-cache/native-inputs/a.png ~]\n\n' +
      'What two colors?'

    const stored = [
      { role: 'user', content: 'ALPHA please', text: 'ALPHA please', row_id: 11 },
      { role: 'assistant', content: 'ALPHA', text: 'ALPHA', row_id: 12 },
      { role: 'user', content: vision, text: vision, row_id: 13 },
      { role: 'assistant', content: 'Blue and red.', text: 'Blue and red.', row_id: 14 }
    ]

    try {
      const { composer, slash, sys } = slashHarness(stored)

      slash('/rewind')
      await flush()
      const browse = $overlayState.get().rewind!

      browse.onEdit(browse.targets[1]!)
      expect(composer.attachImagePaths).not.toHaveBeenCalled()
      // The model still gets the image the turn named: nothing is silently dropped.
      expect(composer.setInput).toHaveBeenCalledWith(vision)
      expect(sys.mock.calls.map(([text]) => String(text)).join('\n')).toContain('could not be re-attached')
    } finally {
      vi.unstubAllEnvs()
    }
  })
})

describe('native pager paging', () => {
  it('PgUp/PgDn move one screen, ↑/↓ one row', async () => {
    const many = Array.from({ length: 30 }, (_, i) => [
      { role: 'user' as const, text: `PROMPT ${i + 1}` },
      { role: 'assistant' as const, text: `reply ${i + 1}` }
    ]).flat()

    const handlers = {
      onBack: vi.fn(),
      onClose: vi.fn(),
      onEdit: vi.fn(),
      onFork: vi.fn(),
      onPaste: vi.fn(),
      onType: vi.fn()
    }
    const manyTargets = Array.from({ length: 30 }, (_, i) => ({
      rowId: 100 + 2 * i,
      text: `PROMPT ${i + 1}`,
      turn: i + 1
    }))
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 24 })
    const frames: string[] = []
    stdout.on('data', chunk => frames.push(stripAnsi(chunk.toString())))

    const instance = renderSync(
      <RewindBrowse
        cols={100}
        native
        {...handlers}
        req={{ messages: many, onSubmit: vi.fn(), targets: manyTargets, ...handlers }}
        t={DEFAULT_THEME}
      />,
      { patchConsole: false, stdin: stdin as any, stdout: stdout as any }
    )

    // The topmost pager row on screen, as `› PROMPT n` / `• reply n` / blank → its line number in the pager.
    const lineOf = (text: string) => {
      const prompt = /› PROMPT (\d+)/.exec(text)
      const reply = /• reply (\d+)/.exec(text)

      return prompt ? (Number(prompt[1]) - 1) * 4 : reply ? (Number(reply[1]) - 1) * 4 + 2 : -1
    }

    const top = () => {
      const screen = frames.join('')
      const hits = [...screen.matchAll(/(?:› PROMPT|• reply) \d+/g)].map(hit => lineOf(hit[0]))

      frames.length = 0

      return Math.min(...hits)
    }

    await settle()
    const bottom = top()
    const rows = 24 - 8

    stdin.send('\x1b[5~') // PgUp
    await settle(80)
    const paged = top()

    expect(bottom - paged).toBeGreaterThanOrEqual(rows - 1)
    expect(bottom - paged).toBeLessThanOrEqual(rows + 1)
    stdin.send('\x1b[A')
    await settle(80)
    expect(paged - top()).toBeLessThanOrEqual(2)
    stdin.send('\x1b[6~') // PgDn
    await settle(80)
    expect(top()).toBeGreaterThan(paged)
    instance.unmount()
  })
})
