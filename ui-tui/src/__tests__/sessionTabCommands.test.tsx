import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { openSessionTab, resetSessionTabs } from '../app/sessionTabsStore.js'
import { withSessionTabCommands } from '../app/slashCatalog.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useSubmission } from '../app/useSubmission.js'

afterEach(() => {
  resetUiState()
  resetSessionTabs()
  resetOverlayState()
})

function fixture() {
  patchUiState({ sid: 'a', busy: true, info: { cwd: '/project', model: 'test', skills: {}, tools: {} } })
  openSessionTab({ id: 'a', cwd: '/project', title: 'First' })
  openSessionTab({ id: 'b', cwd: '/project', title: 'Second' })
  openSessionTab({ id: 'foreign', cwd: '/project-other', title: 'Foreign' })
  const request = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({}))

  const tabAction = vi.fn(),
    newLiveSession = vi.fn(),
    newSession = vi.fn(),
    appendMessage = vi.fn()

  const stage = vi.fn(),
    enqueue = vi.fn(),
    sys = vi.fn(),
    panel = vi.fn()

  const ctx = {
    gateway: { gw: { isCanonical: true, request } },
    local: {},
    session: { tabAction, newLiveSession, newSession, guardBusySessionSwitch: vi.fn(() => true) },
    transcript: { send: vi.fn(), sys, panel, page: vi.fn() },
    slashFlightRef: { current: 0 }
  } as any

  const slash = createSlashHandler(ctx)
  let submission!: ReturnType<typeof useSubmission>

  function Harness() {
    submission = useSubmission({
      appendMessage,
      composerActions: { pushHistory: vi.fn(), clearIn: vi.fn(), stage, enqueue } as any,
      composerRefs: { tokensRef: { current: [] }, queueRef: { current: [] }, queueEditRef: { current: null } } as any,
      composerState: { input: '', inputBuf: [], completions: [] } as any,
      gw: ctx.gateway.gw,
      setLastUserMsg: vi.fn(),
      slashRef: { current: slash },
      submitRef: { current: vi.fn() },
      sys
    })

    return <Text>tab command test</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 30, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  return {
    get submission() {
      return submission
    },
    ctx,
    newLiveSession,
    newSession,
    tabAction,
    stage,
    enqueue,
    request,
    appendMessage,
    sys,
    panel,
    close: () => instance.unmount()
  }
}

it.each(['queue', 'steer', 'interrupt'] as const)(
  'never sends /new as a model prompt, correction, queue entry or transcript message in %s mode',
  async mode => {
    const f = fixture()

    try {
      patchUiState({ busyInputMode: mode })
      f.submission.dispatchSubmission('  /new Review changes')
      await new Promise(resolve => setImmediate(resolve))
      expect(f.newLiveSession).toHaveBeenCalledWith(undefined, 'Review changes')
      expect(f.newSession).not.toHaveBeenCalled()
      expect(f.stage).not.toHaveBeenCalled()
      expect(f.enqueue).not.toHaveBeenCalled()
      expect(f.appendMessage).not.toHaveBeenCalled()
      expect(
        f.request.mock.calls.every(([method]) => ['shared_metrics.slash_command', 'command.observe'].includes(method))
      ).toBe(true)
    } finally {
      f.close()
    }
  }
)

it.each([
  ['/tab 2', { kind: 'index', index: 1 }],
  ['/tab next', { kind: 'cycle', direction: 1 }],
  ['/next', { kind: 'cycle', direction: 1 }],
  ['/prev', { kind: 'cycle', direction: -1 }],
  ['/tabs Second', { kind: 'select', id: 'b' }],
  ['/tabclose', { kind: 'close' }]
])('routes %s to the existing tab controller while a turn is running', (command, action) => {
  const f = fixture()

  try {
    f.submission.dispatchSubmission(command as string)
    expect(f.tabAction).toHaveBeenCalledWith(action)
    expect(f.stage).not.toHaveBeenCalled()
  } finally {
    f.close()
  }
})

it('lists only this directory and refuses invalid indices without a gateway fallback', () => {
  const f = fixture()

  try {
    f.submission.dispatchSubmission('/tabs')
    expect(JSON.stringify(f.panel.mock.calls)).toContain('First')
    expect(JSON.stringify(f.panel.mock.calls)).not.toContain('Foreign')
    f.submission.dispatchSubmission('/tab 3')
    expect(f.sys).toHaveBeenCalledWith(expect.stringContaining('No tab 3'))
    expect(f.tabAction).not.toHaveBeenCalled()
  } finally {
    f.close()
  }
})

it('includes local tab commands and subcommands in completion/help without aliasing /new back to /clear', () => {
  const catalog = withSessionTabCommands({
    pairs: [],
    canon: { '/new': '/clear' },
    categories: [],
    sub: {},
    skillCount: 0
  })

  expect(catalog.canon['/new']).toBe('/new')
  expect(catalog.pairs.some(([name]) => name === '/tab')).toBe(true)
  expect(catalog.sub['/tab']).toContain('next')
})
