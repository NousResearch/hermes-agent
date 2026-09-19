import { PassThrough } from 'node:stream'

import { Box, renderSync } from '@hermes/ink'
import type { PluginCardActionResult } from '@hermes/shared/gateway-events'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { OverlayState } from '../app/interfaces.js'
import {
  clearPluginNoticeForSession,
  dismissPluginNotice,
  getOverlayState,
  patchOverlayState,
  resetOverlayState
} from '../app/overlayStore.js'
import { PromptZone } from '../components/appOverlays.js'
import { isPluginNoticeCancelInput, PluginNoticePrompt } from '../components/prompts.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'

const notice = () => ({
  actions: [
    { args: 'share', command: 'skill-choice', label: 'Share' },
    { args: 'portal', command: 'skill-choice', label: 'Review in Portal' },
    { args: 'later', command: 'skill-choice', label: 'Maybe Later' }
  ],
  body: 'Why it qualified\nSynthetic qualification evidence.\nNothing has been uploaded or shared.',
  plugin_id: 'team-tools',
  plugin_name: 'Team Tools',
  title: 'Synthetic checklist looks ready to share with your team.'
})

function mount(
  onAction: (command: string, args: string) => Promise<PluginCardActionResult>,
  onCancel = vi.fn(),
  onReplace = vi.fn(() => true),
  onResult = vi.fn(() => true)
) {
  const stdout = Object.assign(new PassThrough(), { columns: 110, rows: 36, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let output = ''
  stdout.on('data', chunk => (output += stripAnsi(chunk.toString())))

  const view = renderSync(
    <Box height={36}>
      <PluginNoticePrompt
        cols={110}
        notice={notice()}
        onAction={onAction}
        onCancel={onCancel}
        onReplace={onReplace}
        onResult={onResult}
        t={DEFAULT_THEME}
      />
    </Box>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  return {
    cleanup: () => {
      view.unmount()
      view.cleanup()
    },
    input: (value: string) => stdin.write(value),
    frame: () => {
      const start = output.lastIndexOf('╔')
      const end = output.lastIndexOf('╝')

      return start >= 0 && end >= start ? output.slice(start, end + 1) : output
    },
    onCancel,
    onReplace,
    output: () => output
  }
}

it('shows direct choices, replaces the menu, blocks duplicate submits, reports errors, and dismisses', async () => {
  let reject!: (error: Error) => void

  const onAction = vi
    .fn()
    .mockImplementationOnce(() => new Promise((_resolve, rej) => (reject = rej)))
    .mockResolvedValueOnce({
      card: {
        actions: [{ args: 'ack', command: 'skill-choice', label: 'Acknowledge' }],
        body: Array.from({ length: 40 }, (_, index) => `REVIEW-${String(index + 1).padStart(2, '0')}`).join('\n'),
        plugin_id: 'team-tools',
        plugin_name: 'Team Tools',
        title: 'Team update available'
      },
      kind: 'card'
    })

  const onCancel = vi.fn()
  const view = mount(onAction, onCancel)

  try {
    expect(view.output()).toContain('Synthetic checklist looks ready to share with your team.')
    expect(view.output()).toContain('Why it qualified')
    expect(view.output()).toContain('Nothing has been uploaded or shared.')
    expect(view.output()).toContain('Share')
    expect(view.output()).toContain('Review in Portal')
    expect(view.output()).toContain('Maybe Later')
    expect(view.output()).not.toContain('Ctrl+O')
    view.input('\x1b[B')
    view.input('\r')
    await vi.waitFor(() => expect(onAction).toHaveBeenCalledWith('skill-choice', 'portal'))
    view.input('\r')
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(onAction).toHaveBeenCalledTimes(1)
    reject(new Error('plugin went away'))
    await vi.waitFor(() => expect(view.output()).toContain('plugin went away'))
    view.input('\r')
    await vi.waitFor(() => expect(onAction).toHaveBeenCalledTimes(2))
    await vi.waitFor(() => expect(view.onReplace).toHaveBeenCalledWith(expect.objectContaining({ title: 'Team update available' })))
    await vi.waitFor(() => expect(view.output()).toContain('Team update available'))
    await vi.waitFor(() => expect(view.output()).toContain('Acknowledge'))
    expect(view.frame()).toContain('REVIEW-01')
    expect(view.frame()).not.toContain('REVIEW-40')
    expect(view.frame()).toContain('Full notice shown above')
    expect(view.frame()).toContain('Acknowledge')
    expect(view.frame().split('\n').length).toBeLessThanOrEqual(36)
    view.input('\x1b')
    await vi.waitFor(() => expect(onCancel).toHaveBeenCalledTimes(1))
  } finally {
    view.cleanup()
  }
})

it('does not let a late completion dismiss a newer notice and clears notices on session change', () => {
  resetOverlayState()
  const oldNotice = notice()
  const nextNotice = { ...notice(), title: 'Newer notice' }

  patchOverlayState({ pluginNotice: { notice: oldNotice, sessionId: 'sid-a' } })
  patchOverlayState({ pluginNotice: { notice: nextNotice, sessionId: 'sid-a' } })
  expect(dismissPluginNotice(oldNotice)).toBe(false)
  expect(getOverlayState().pluginNotice?.notice).toBe(nextNotice)

  clearPluginNoticeForSession('sid-b')
  expect(getOverlayState().pluginNotice).toBeNull()
})

function mountPromptZone(request: ReturnType<typeof vi.fn>, onPluginResult = vi.fn()) {
  const stdout = Object.assign(new PassThrough(), { columns: 110, rows: 36, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let output = ''
  stdout.on('data', chunk => (output += stripAnsi(chunk.toString())))

  const gateway = {
    gw: { request, send: () => {} } as unknown as GatewayClient,
    rpc: vi.fn()
  }

  const view = renderSync(
    <GatewayProvider value={gateway}>
      <PromptZone
        cols={110}
        onApprovalChoice={vi.fn()}
        onClarifyAnswer={vi.fn()}
        onClarifyQuestionAnswer={vi.fn()}
        onPluginResult={onPluginResult}
        onSecretSubmit={vi.fn()}
        onSudoSubmit={vi.fn()}
        onVaultUnlockSubmit={vi.fn()}
      />
    </GatewayProvider>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  return {
    cleanup: () => {
      view.unmount()
      view.cleanup()
    },
    input: (value: string) => stdin.write(value),
    onPluginResult,
    output: () => output
  }
}

it.each([
  ['sudo', { sudo: { requestId: 'sudo-1' } }, 'sudo password required'],
  ['secret', { secret: { envVar: 'API_TOKEN', prompt: 'Enter API token', requestId: 'secret-1' } }, 'Enter API token'],
  [
    'vault unlock',
    { vaultUnlock: { displayName: '1Password', requestId: 'vault-1' } },
    'Unlock 1Password for this session'
  ]
])('keeps the %s security prompt ahead of a plugin notice', (_name, securityOverlay, expected) => {
  resetOverlayState()
  patchOverlayState({ pluginNotice: { notice: notice(), sessionId: 'sid-a' }, ...securityOverlay } as Partial<OverlayState>)
  const view = mountPromptZone(vi.fn())

  try {
    expect(view.output()).toContain(expected)
    expect(view.output()).not.toContain('Synthetic checklist looks ready to share with your team.')
  } finally {
    view.cleanup()
    resetOverlayState()
  }
})

it('projects the full action-returned card once and ignores a result after dismissal', async () => {
  resetOverlayState()

  const initial = notice()

  const replacement = {
    ...notice(),
    actions: [{ args: 'confirm', command: 'skill-choice', label: 'Confirm' }],
    body: Array.from({ length: 40 }, (_, index) => `REVIEW-${String(index + 1).padStart(2, '0')}`).join('\n'),
    title: 'Full review'
  }

  let resolve!: (result: PluginCardActionResult) => void
  const request = vi.fn(() => new Promise<PluginCardActionResult>(done => (resolve = done)))
  patchOverlayState({ pluginNotice: { notice: initial, sessionId: 'sid-a' } })
  const view = mountPromptZone(request)

  try {
    view.input('\r')
    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    resolve({ card: replacement, kind: 'card' })
    await vi.waitFor(() => expect(view.onPluginResult).toHaveBeenCalledWith('Team Tools · Full review', replacement.body))
    expect(getOverlayState().pluginNotice?.notice).toBe(replacement)

    view.input('\r')
    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(2))
    dismissPluginNotice(replacement)
    resolve({ card: { ...replacement, title: 'Too late' }, kind: 'card' })
    await new Promise(done => setTimeout(done, 20))
    expect(view.onPluginResult).toHaveBeenCalledTimes(1)
    expect(getOverlayState().pluginNotice).toBeNull()
  } finally {
    view.cleanup()
    resetOverlayState()
  }
})

it('projects a successful text result, dismisses the completed notice, and cannot repeat the action', async () => {
  const onAction = vi.fn().mockResolvedValue({ kind: 'text', text: 'Shared successfully.' })
  const onCancel = vi.fn()
  const onResult = vi.fn(() => true)
  const view = mount(onAction, onCancel, vi.fn(), onResult)

  try {
    view.input('\r')
    await vi.waitFor(() => expect(onResult).toHaveBeenCalledWith('Shared successfully.'))
    expect(onCancel).toHaveBeenCalledTimes(1)

    view.input('\r')
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(onAction).toHaveBeenCalledTimes(1)
  } finally {
    view.cleanup()
  }
})

it('isolates a replacement notice from the previous action’s busy state and error', async () => {
  resetOverlayState()
  let rejectOld!: (error: Error) => void

  const request = vi.fn()
    .mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectOld = reject }))
    .mockImplementation(() => new Promise(() => {}))

  patchOverlayState({ pluginNotice: { notice: notice(), sessionId: 'sid-a' } })
  const view = mountPromptZone(request)

  try {
    view.input('\r')
    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    patchOverlayState({ pluginNotice: { notice: { ...notice(), title: 'New proactive notice' }, sessionId: 'sid-a' } })
    await vi.waitFor(() => expect(view.output()).toContain('New proactive notice'))
    view.input('\r')
    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(2))
    rejectOld(new Error('Obsolete action error'))
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(view.output()).not.toContain('Obsolete action error')
    view.input('\r')
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(request).toHaveBeenCalledTimes(2)
    expect(view.onPluginResult).not.toHaveBeenCalled()
  } finally {
    view.cleanup()
    resetOverlayState()
  }
})

it('treats Ctrl+C as plugin notice cancellation', () => {
  expect(isPluginNoticeCancelInput('c', { ctrl: true, escape: false })).toBe(true)
  expect(isPluginNoticeCancelInput('', { ctrl: false, escape: true })).toBe(true)
  expect(isPluginNoticeCancelInput('c', { ctrl: false, escape: false })).toBe(false)
})