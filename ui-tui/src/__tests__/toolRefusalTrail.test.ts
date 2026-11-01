import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: { dequeue: () => undefined, queueEditRef: ref<null | number>(null), sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: { appendMessage: (msg: Msg) => appended.push(msg), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

// tui_gateway/tool_progress.py::_on_tool_complete sends the parsed terminal envelope as `result`;
// this is terminal_tool._error_json for an approval that timed out on the gateway path.
const TIMED_OUT = {
  output: '',
  exit_code: -1,
  error: 'BLOCKED: Command timed out without user response. Do NOT retry this command.',
  status: 'blocked',
  user_summary: 'No answer within 20 seconds — the command did not run.'
}

const trailFor = (command: string, result: unknown): string => {
  const appended: Msg[] = []
  const onEvent = createGatewayEventHandler(buildCtx(appended))
  const emit = (type: string, payload: any) => onEvent({ type, payload, session_id: 'focused' } as any)

  patchUiState({ sid: 'focused', info: { model: 'test', skills: {}, tools: {} } as any })
  emit('message.start', {})
  emit('tool.start', { tool_id: 'c1', name: 'terminal', context: command })
  emit('tool.complete', { tool_id: 'c1', name: 'terminal', duration_s: 0.1, result })
  emit('message.complete', { text: 'done' })

  return appended.find(m => m.kind === 'trail')?.tools?.find(line => line.includes(command)) ?? ''
}

describe('a refused command in the tool trail', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('says it did not run and is marked ✗, not ✓', () => {
    const line = trailFor('rm -rf /srv/data', TIMED_OUT)

    expect(line).toContain('did not run')
    expect(line.endsWith(' ✗')).toBe(true)
  })

  it('marks a lifecycle-guard refusal ✗ with its reason', () => {
    // terminal_tool_guards.gateway_lifecycle_block: status "error", no user_summary.
    const line = trailFor('hermes gateway restart', {
      output: '',
      exit_code: 1,
      error: 'Blocked: command or referenced script cannot restart, stop, or uninstall the gateway.',
      status: 'error'
    })

    expect(line).toContain('cannot restart')
    expect(line.endsWith(' ✗')).toBe(true)
  })

  it('leaves a command that ran marked ✓', () => {
    expect(trailFor('ls /srv', { output: 'data', exit_code: 0 }).endsWith(' ✓')).toBe(true)
  })
})
