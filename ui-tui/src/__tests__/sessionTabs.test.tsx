import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React, { useRef, useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { resetOverlayState } from '../app/overlayStore.js'
import {
  $sessionTabs,
  findSessionTab,
  openSessionTab,
  refreshSessionTabs,
  resetSessionTabs,
  saveSessionTabDraft,
  tabsInDirectory
} from '../app/sessionTabsStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { useSessionTabDrafts, useSessionTabs } from '../app/useSessionTabs.js'
import { SessionTabsBar, sessionTabWindow } from '../components/sessionTabsBar.js'
import { shouldPassThroughToGlobalHandler } from '../components/textInput.js'
import { sessionTabShortcut } from '../domain/sessionTabs.js'
import { DEFAULT_THEME } from '../theme.js'

const mountOptions = () => ({
  stdin: Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() }) as any,
  stdout: Object.assign(new PassThrough(), { columns: 120, rows: 35, isTTY: false }) as any,
  stderr: new PassThrough() as any,
  patchConsole: false
})

afterEach(() => {
  resetSessionTabs()
  resetUiState()
  resetOverlayState()
  vi.unstubAllEnvs()
})

it('stores only opened views, merges resume aliases, and scopes by exact directory', () => {
  openSessionTab({ id: 'a', storedId: 'old-a', cwd: '/projects/Alpha', title: 'Alpha chat' })
  saveSessionTabDraft('a', { input: 'draft', inputBuf: ['line'], tokens: [] })
  openSessionTab({
    id: 'canonical-a',
    requestedId: 'old-a',
    storedId: 'new-a',
    cwd: '/projects/Alpha',
    title: 'Renamed'
  })
  openSessionTab({ id: 'other', cwd: '/projects/Alpha-other', title: 'Not this folder' })
  openSessionTab({ id: 'child', cwd: '/projects/Alpha/app', title: 'Not this directory' })
  expect(tabsInDirectory($sessionTabs.get(), '/projects/Alpha/').map(x => x.id)).toEqual(['canonical-a'])
  expect(findSessionTab('old-a')?.draft.input).toBe('draft')
  expect(findSessionTab('canonical-a')?.aliases).toContain('a')
  expect(findSessionTab('canonical-a')?.title).toBe('Renamed')
})

it('updates opened titles and running badges without importing the global roster into tabs', () => {
  openSessionTab({ id: 'a', cwd: '/projects/Alpha' })
  refreshSessionTabs([
    { id: 'a', title: 'Real title', running: true },
    { id: 'foreign', title: 'Do not open this' }
  ])
  expect($sessionTabs.get()).toHaveLength(1)
  expect(findSessionTab('a')).toMatchObject({ title: 'Real title', running: true })
})

it('keeps the active tab visible when the strip overflows narrow terminals', () => {
  for (const columns of [24, 40, 80, 160]) {
    for (let active = 0; active < 30; active++) {
      const w = sessionTabWindow(30, active, columns)
      expect(w.offset).toBeLessThanOrEqual(active)
      expect(w.offset + w.count).toBeGreaterThan(active)
      expect(w.width * w.count + 9).toBeLessThanOrEqual(columns)
    }
  }
})

it('renders clean, directory-scoped tab titles with new and close controls', async () => {
  openSessionTab({ id: 'a', cwd: '/projects/Alpha', title: '[OpenCode] NFT swap [123456ABCD]' })
  openSessionTab({ id: 'b', cwd: '/projects/Alpha', title: 'Traits', running: true })
  openSessionTab({ id: 'foreign', cwd: '/projects/Beta', title: 'FOREIGN BETA' })
  const opts = mountOptions()
  let output = ''
  opts.stdout.on('data', (chunk: Buffer) => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <SessionTabsBar columns={120} currentId="a" cwd="/projects/Alpha" onAction={vi.fn()} t={DEFAULT_THEME} />,
    opts
  )

  try {
    await vi.waitFor(() => expect(output).toContain('NFT swap'))
    expect(output).toContain('Traits')
    expect(output).toContain('+')
    expect(output).toContain('×')
    expect(output).not.toContain('[OpenCode]')
    expect(output).not.toContain('123456ABCD')
    expect(output).not.toContain('FOREIGN BETA')
  } finally {
    instance.unmount()
  }
})

it('adds tab shortcuts without hijacking autocomplete, agents, or delete-word', () => {
  expect(sessionTabShortcut('n', { meta: true })).toEqual({ kind: 'new' })
  expect(sessionTabShortcut('w', { meta: true })).toEqual({ kind: 'close' })
  expect(sessionTabShortcut('3', { meta: true })).toEqual({ kind: 'index', index: 2 })
  expect(sessionTabShortcut('', { ctrl: true, pageDown: true })).toEqual({ kind: 'cycle', direction: 1 })
  expect(sessionTabShortcut('', { meta: true, rightArrow: true })).toBeUndefined()
  expect(sessionTabShortcut('', { ctrl: true, tab: true, shift: true })).toEqual({ kind: 'cycle', direction: -1 })
  expect(sessionTabShortcut('', { tab: true })).toBeUndefined()
  expect(sessionTabShortcut('t', { ctrl: true })).toBeUndefined()
  expect(sessionTabShortcut('w', { ctrl: true })).toBeUndefined()
  expect(shouldPassThroughToGlobalHandler('n', { meta: true } as any)).toBe(true)
  expect(shouldPassThroughToGlobalHandler('w', { meta: true } as any)).toBe(true)
})

function fixture() {
  vi.stubEnv('HERMES_CWD', '/projects/Alpha')
  resetUiState()
  resetSessionTabs()
  resetOverlayState()
  let serial = 0
  let failActivation = false
  const sys = vi.fn()

  const rows = new Map<string, any>([
    ['a', { title: 'Alpha first', running: false }],
    ['b', { title: 'Alpha second', running: false }]
  ])

  const response = (id: string) => ({
    session_id: id,
    stored_session_id: id,
    subscription_id: 'sub-' + id,
    messages: [],
    started_at: 1,
    running: rows.get(id)?.running || false,
    info: { cwd: '/projects/Alpha', title: rows.get(id)?.title || '', model: 'test', skills: {}, tools: {} }
  })

  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.create') {
      const id = 'new-' + ++serial
      rows.set(id, { title: '', running: false })

      return response(id)
    }

    if (method === 'session.resume' || method === 'session.activate') {
      if (failActivation) {
        throw new Error('activation failed')
      }

      return response(params.session_id)
    }

    if (method === 'session.detach') {
      return { detached: true, ...params }
    }

    throw new Error('Unexpected method ' + method)
  })

  let life!: ReturnType<typeof useSessionLifecycle>
  let action!: ReturnType<typeof useSessionTabs>
  let composer: any

  function Harness() {
    const [input, setText] = useState('')
    const [inputBuf, setBuf] = useState<string[]>([])
    const [tokens, setTokens] = useState<any[]>([])
    const inputRef = useRef('')
    const inputBufRef = useRef<string[]>([])
    const tokensRef = useRef<any[]>([])
    const historyDraftRef = useRef('')

    const setInput = (next: any) => {
      inputRef.current = typeof next === 'function' ? next(inputRef.current) : next
      setText(inputRef.current)
    }

    const setInputBuf = (next: any) => {
      inputBufRef.current = typeof next === 'function' ? next(inputBufRef.current) : next
      setBuf(inputBufRef.current)
    }

    const setComposerTokens = (next: any) => {
      tokensRef.current = typeof next === 'function' ? next(tokensRef.current) : next
      setTokens(tokensRef.current)
    }

    composer = {
      state: { input, inputBuf, tokens },
      refs: { inputRef, inputBufRef, tokensRef, historyDraftRef },
      actions: { setInput, setInputBuf, setComposerTokens, setQueueEdit: vi.fn(), setHistoryIdx: vi.fn() }
    }
    const drafts = useSessionTabDrafts(composer)
    life = useSessionLifecycle({
      colsRef: { current: 120 },
      composerActions: composer.actions,
      gw: { isCanonical: true, request } as any,
      rpc: request as any,
      scrollRef: { current: null },
      panel: vi.fn(),
      sys,
      setHistoryItems: vi.fn(),
      setLastUserMsg: vi.fn(),
      setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn(),
      onBeforeSessionChange: drafts.capture,
      onSessionActivated: drafts.activate
    })
    action = useSessionTabs(life, drafts.capture, sys)

    return <Text>{input}</Text>
  }

  const instance = renderSync(<Harness />, mountOptions())

  return {
    request,
    sys,
    instance,
    rows,
    life: () => life,
    action: (value: any) => action(value),
    composer: () => composer,
    fail: () => {
      failActivation = true
    },
    async attach(id: string) {
      await life.resumeById(id)
      await vi.waitFor(() => expect(getUiState().sid).toBe(id))
    },
    cleanup: () => instance.unmount()
  }
}

it('preserves per-tab text, multiline and paste/image tokens across new and switch', async () => {
  const f = fixture()

  try {
    await f.attach('a')
    const token = { kind: 'paste', label: '[[ Paste 1 ]]', text: 'original pasted contents' }
    f.composer().actions.setInput('draft A [[ Paste 1 ]]')
    f.composer().actions.setInputBuf(['earlier line'])
    f.composer().actions.setComposerTokens([token])
    f.action({ kind: 'new' })
    await vi.waitFor(() => expect(getUiState().sid).toBe('new-1'))
    expect(f.composer().refs.inputRef.current).toBe('')
    f.composer().actions.setInput('draft B')
    f.action({ kind: 'index', index: 0 })
    await vi.waitFor(() => expect(getUiState().sid).toBe('a'))
    expect(f.composer().refs.inputRef.current).toBe('draft A [[ Paste 1 ]]')
    expect(f.composer().refs.inputBufRef.current).toEqual(['earlier line'])
    expect(f.composer().refs.tokensRef.current).toEqual([token])
    f.action({ kind: 'cycle', direction: 1 })
    await vi.waitFor(() => expect(getUiState().sid).toBe('new-1'))
    expect(f.composer().refs.inputRef.current).toBe('draft B')
  } finally {
    f.cleanup()
  }
})

it('switches while a prior tab runs without interrupting or deleting it', async () => {
  const f = fixture()

  try {
    await f.attach('a')
    patchUiState({ busy: true })
    f.request.mockClear()
    f.action({ kind: 'new' })
    await vi.waitFor(() => expect(getUiState().sid).toBe('new-1'))
    expect(findSessionTab('a')).toBeDefined()
    const methods = f.request.mock.calls.map(([name]) => name)
    expect(methods).not.toContain('session.interrupt')
    expect(methods).not.toContain('session.mutate')
    expect(methods).not.toContain('session.close')
    expect(
      f.request.mock.calls.filter(([name]) => name === 'session.detach').every(([, params]) => !params.end_reason)
    ).toBe(true)
  } finally {
    f.cleanup()
  }
})

it('closing a tab retains its history and refuses to discard an unsent draft', async () => {
  const f = fixture()

  try {
    await f.attach('a')
    f.composer().actions.setInput('keep me')
    f.action({ kind: 'close' })
    await vi.waitFor(() => expect(f.sys).toHaveBeenCalledWith(expect.stringContaining('unsent draft')))
    expect(findSessionTab('a')).toBeDefined()
    f.composer().actions.setInput('')
    f.action({ kind: 'new' })
    await vi.waitFor(() => expect(getUiState().sid).toBe('new-1'))
    f.request.mockClear()
    f.action({ kind: 'close', id: 'a' })
    await vi.waitFor(() => expect(findSessionTab('a')).toBeUndefined())
    expect(f.request).not.toHaveBeenCalled()
    f.action({ kind: 'close' })
    await vi.waitFor(() => expect(getUiState().sid).toBe('new-2'))
    expect(findSessionTab('new-1')).toBeUndefined()
    expect(f.request.mock.calls.map(([name]) => name)).not.toContain('session.mutate')
  } finally {
    f.cleanup()
  }
})

it('a failed switch leaves the current tab and its draft intact', async () => {
  const f = fixture()

  try {
    await f.attach('a')
    await f.attach('b')
    f.composer().actions.setInput('still in B')
    f.fail()
    f.action({ kind: 'select', id: 'a' })
    await vi.waitFor(() => expect(f.sys).toHaveBeenCalledWith('error: activation failed'))
    expect(getUiState().sid).toBe('b')
    expect(f.composer().refs.inputRef.current).toBe('still in B')
    expect($sessionTabs.get()).toHaveLength(2)
  } finally {
    f.cleanup()
  }
})
