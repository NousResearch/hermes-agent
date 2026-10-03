import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import type { ComposerToken } from '../app/interfaces.js'
import { prepareSubmission } from '../app/useSubmission.js'

const cleanups: (() => void)[] = []
beforeEach(() => {
  resetUiState()
  resetTurnState()
  turnController.fullReset()
})
afterEach(() => {
  for (const cleanup of cleanups.splice(0)) cleanup()
})

function mount(request: ReturnType<typeof vi.fn>, initialTokens?: ComposerToken[]) {
  let api: ReturnType<typeof useSessionLifecycle> | undefined
  let tokens: ComposerToken[] = initialTokens ?? [
    { kind: 'paste', label: '[[ Paste 1 ]]', text: 'unsent\nmultiline\ndraft' }
  ]
  const setComposerTokens = vi.fn(next => {
    tokens = typeof next === 'function' ? next(tokens) : next
  })
  function Probe() {
    const lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens } as any,
      gw: { request } as any,
      panel: vi.fn(),
      rpc: vi.fn(async () => null),
      scrollRef: { current: null },
      setHistoryItems: vi.fn(),
      setLastUserMsg: vi.fn(),
      setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn(),
      sys: vi.fn()
    })
    useEffect(() => {
      api = lifecycle
    })
    return null
  }
  const stream = () => Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  const app = renderSync(React.createElement(Probe), {
    patchConsole: false,
    stderr: stream() as any,
    stdin: stream() as any,
    stdout: stream() as any
  })
  cleanups.push(() => app.unmount())
  return {
    api: () => api!,
    tokens: () => tokens,
    setTokens: (next: ComposerToken[]) => {
      tokens = next
    }
  }
}

it('preserves collapsed unsent draft content when automatically recovering the session', async () => {
  const request = vi.fn(async () => ({ session_id: 'live-again', stored_session_id: 'durable', messages: [] }))
  const probe = mount(request)
  await vi.waitFor(() => expect(probe.api()).toBeTruthy())
  const original = probe.tokens()
  await probe.api().resumeById('durable', { preserveTextDraft: true })
  expect(getUiState().sid).toBe('live-again')
  expect(probe.tokens()).toEqual(original)
  expect(prepareSubmission('Please read [[ Paste 1 ]]', probe.tokens()).text).toContain('unsent\nmultiline\ndraft')
  expect(request).toHaveBeenCalledExactlyOnceWith('session.resume', { cols: 80, session_id: 'durable' })
})

it('still clears attachment tokens on an explicit session switch', async () => {
  const probe = mount(vi.fn(async () => ({ session_id: 'other', messages: [] })))
  await vi.waitFor(() => expect(probe.api()).toBeTruthy())
  await probe.api().resumeById('other')
  expect(probe.tokens()).toEqual([])
})

it('keeps edits made while recovery is awaiting the server rather than restoring an old snapshot', async () => {
  let finish!: (value: unknown) => void
  const request = vi.fn(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  const probe = mount(request)
  await vi.waitFor(() => expect(probe.api()).toBeTruthy())
  const recovery = probe.api().resumeById('durable', { preserveTextDraft: true })
  await vi.waitFor(() => expect(request).toHaveBeenCalledOnce())
  probe.setTokens([{ kind: 'paste', label: '[[ Paste 2 ]]', text: 'edited while offline' }])
  finish({ session_id: 'live-again', messages: [] })
  await recovery
  expect(probe.tokens()).toEqual([{ kind: 'paste', label: '[[ Paste 2 ]]', text: 'edited while offline' }])
  expect(prepareSubmission('[[ Paste 2 ]]', probe.tokens()).text).toBe('edited while offline')
})

it('recovers empty drafts without inventing content', async () => {
  const probe = mount(
    vi.fn(async () => ({ session_id: 'live-again', messages: [] })),
    []
  )
  await vi.waitFor(() => expect(probe.api()).toBeTruthy())
  await probe.api().resumeById('durable', { preserveTextDraft: true })
  expect(probe.tokens()).toEqual([])
})

it('does not restore server-owned image attachments as text drafts', async () => {
  const probe = mount(
    vi.fn(async () => ({ session_id: 'live-again', messages: [] })),
    [
      { kind: 'image', index: 1, label: '[[ Image 1 ]]', path: '/fixture/image.png' },
      { kind: 'paste', label: '[[ Paste 1 ]]', text: 'keep me' }
    ]
  )
  await vi.waitFor(() => expect(probe.api()).toBeTruthy())
  await probe.api().resumeById('durable', { preserveTextDraft: true })
  expect(probe.tokens()).toEqual([{ kind: 'paste', label: '[[ Paste 1 ]]', text: 'keep me' }])
})

it('keeps the draft when recovery fails before hydration', async () => {
  const probe = mount(
    vi.fn(async () => {
      throw new Error('offline')
    })
  )
  await vi.waitFor(() => expect(probe.api()).toBeTruthy())
  const original = probe.tokens()
  await probe.api().resumeById('durable', { preserveTextDraft: true })
  expect(probe.tokens()).toEqual(original)
  expect(getUiState().sid).toBeNull()
})
