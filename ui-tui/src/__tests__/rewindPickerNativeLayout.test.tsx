import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { AppLayoutProps } from '../app/interfaces.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_VOICE_RECORD_KEY } from '../lib/platform.js'
import { DEFAULT_THEME } from '../theme.js'

// Native-scrollback mode writes the whole layout into the terminal, so a picker taller than the screen
// pushes its own heading and the selected prompt into scrollback. Mount the real AppLayout at 80x24
// with eight long prompts and a long reply, and read back every row the frame occupies.

const T0 = 1_800_000_000_000

const gatewayStub = {
  gw: {
    request: () => new Promise<never>(() => {}),
    send: () => {}
  } as unknown as GatewayClient,
  rpc: (() => new Promise<never>(() => {})) as never
}

const layoutProps: AppLayoutProps = {
  actions: {
    activateLiveSession: () => {},
    answerApproval: () => {},
    answerSecret: () => {},
    answerSudo: () => {},
    cancelClarify: () => {},
    clearSelection: () => {},
    closeLiveSession: () => Promise.resolve(null),
    newLiveSession: () => {},
    newPromptSession: () => {},
    onModelSelect: () => {},
    resumeById: () => {},
    setStickyPrompt: () => {}
  },
  composer: {
    cols: 120,
    compIdx: 0,
    completions: [],
    empty: true,
    handleTextPaste: () => null,
    input: '',
    inputBuf: [],
    pagerPageSize: 10,
    queueEditIdx: null,
    queuedDisplay: [],
    submit: () => {},
    updateInput: () => {},
    voiceRecordKey: DEFAULT_VOICE_RECORD_KEY
  },
  mouseTracking: 'off',
  progress: { showProgressArea: false },
  status: {
    cwdLabel: '~/repo',
    goodVibesTick: 0,
    lastTurnEndedAt: T0 - 5_000,
    sessionStartedAt: T0 - 60_000,
    sessionTitle: '',
    showStickyPrompt: false,
    statusColor: DEFAULT_THEME.color.ok,
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
      measureRef: () => () => {},
      offsets: [],
      start: 0,
      topSpacer: 0
    },
    virtualRows: []
  }
}


// Components read the terminal height from `process.stdout` (useStdout), so the test sizes that.
const realRows = Object.getOwnPropertyDescriptor(process.stdout, 'rows')

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()

  if (realRows) {
    Object.defineProperty(process.stdout, 'rows', realRows)
  } else {
    delete (process.stdout as { rows?: number }).rows
  }
})

/** The first frame native mode writes for `targets` at 80x24, top to bottom. */
async function firstFrame(targets: Array<{ reply: string; rowId: number; text: string; turn: number }>) {
  vi.stubEnv('HERMES_TUI_NATIVE', '1')
  vi.resetModules()
  Object.defineProperty(process.stdout, 'rows', { configurable: true, value: 24 })
  const { AppLayout } = await import('../components/appLayout.js')
  const { GatewayProvider } = await import('../app/gatewayContext.js')
  const { patchOverlayState, resetOverlayState } = await import('../app/overlayStore.js')
  const { patchUiState, resetUiState } = await import('../app/uiStore.js')

  resetOverlayState()
  resetUiState()
  patchUiState({ sessionTitle: 'test', sid: 'sid-1', status: 'ready' })
  patchOverlayState({ rewind: { onFork: vi.fn(), onSubmit: vi.fn(), targets } })

  const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: vi.fn(), setRawMode: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  const out = { text: '' }

  stdout.on('data', chunk => {
    out.text += chunk.toString()
  })

  const props = { ...layoutProps, composer: { ...layoutProps.composer, cols: 80 } }

  const instance = renderSync(
    <GatewayProvider value={gatewayStub}>
      <AppLayout {...props} />
    </GatewayProvider>,
    { patchConsole: false, stderr: new PassThrough() as any, stdin: stdin as any, stdout: stdout as any }
  )

  try {
    await vi.waitFor(() => expect(stripAnsi(out.text)).toContain('~/repo'))

    // The first frame of a primary-buffer render is the whole layout, top to bottom.
    return stripAnsi(out.text).split('~/repo')[0]!.replace(/^\n/, '').split('\n')
  } finally {
    instance.unmount()
    instance.cleanup()
    resetOverlayState()
    resetUiState()
  }
}

const reply = Array.from({ length: 40 }, (_, i) => `reply row ${i + 1}`).join('\n')

describe('native rewind picker at 80x24', () => {
  it('fits the whole layout with the heading, the selection and the selected exchange on screen', async () => {
    const frame = await firstFrame(
      Array.from({ length: 8 }, (_, i) => ({
        reply,
        rowId: i + 1,
        text: `prompt ${i + 1} long enough to fill the whole picker row at eighty columns and more`,
        turn: i + 1
      }))
    )

    expect(frame.length).toBeLessThanOrEqual(24)
    expect(frame.some(row => row.includes('↺ rewind'))).toBe(true)
    expect(frame.some(row => row.includes('▸ 8. prompt 8'))).toBe(true)
    expect(frame.some(row => row.includes('Hermes replied:'))).toBe(true)
  })

  it('shows the first reply row even when the selected prompt fills the context window', async () => {
    const prompt = Array.from({ length: 6 }, (_, i) => `prompt line ${i + 1}`).join('\n')
    const frame = await firstFrame([{ reply, rowId: 1, text: prompt, turn: 1 }])

    expect(frame.length).toBeLessThanOrEqual(24)
    expect(frame.some(row => row.includes('prompt line 1'))).toBe(true)
    expect(frame.some(row => row.includes('reply row 1'))).toBe(true)
  })
})
