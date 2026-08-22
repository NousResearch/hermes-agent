import { PassThrough } from 'stream'

import { Box, renderSync, stringWidth } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { clearSpawnHistory, getSpawnHistory } from '../app/spawnHistoryStore.js'
import { markSubmitting } from '../app/submissionCore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, patchTurnState, resetTurnState } from '../app/turnStore.js'
import { AgentDock, AgentDockView } from '../components/agentDock.js'
import { AgentsOverlay } from '../components/agentsOverlay.js'
import { projectAgentDock } from '../lib/agentDock.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg, SubagentProgress } from '../types.js'

const NOW = 1_750_000_000_000

const makeItem = (overrides: Partial<SubagentProgress> & Pick<SubagentProgress, 'id' | 'index'>): SubagentProgress => ({
  depth: 0,
  goal: overrides.id,
  notes: [],
  parentId: null,
  status: 'running',
  taskCount: 1,
  thinking: [],
  toolCount: 0,
  tools: [],
  ...overrides
})

const makeStreams = (columns = 100, rows = 24) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns, isTTY: false, rows })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  // Ink's non-TTY path can emit more than one full frame into the stream.
  // Prefer the final complete frame so width/line assertions are independent
  // of NODE_ENV-driven render scheduling (test vs production).
  const getOutput = () => {
    const text = stripAnsi(output)
    const frames = text.split(/\n(?=╭)/).filter(frame => frame.includes('╭') && frame.includes('╰'))

    return frames.at(-1) ?? text
  }

  return { getOutput, stderr, stdin, stdout }
}

const ref = <T,>(current: T) => ({ current })

const buildFlowCtx = (appended: Msg[]) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: {
      gw: { request: vi.fn() },
      rpc: vi.fn(async () => null)
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: ref(vi.fn()) },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: vi.fn(),
      setHistoryItems: vi.fn()
    },
    voice: {
      setProcessing: vi.fn(),
      setRecording: vi.fn(),
      setVoiceEnabled: vi.fn()
    }
  }) as any

const renderView = (subagents: SubagentProgress[], cols = 100) => {
  const streams = makeStreams(cols)

  const instance = renderSync(
    <Box flexDirection="column" width={cols}>
      <AgentDockView cols={cols} nowMs={NOW} onOpen={() => {}} subagents={subagents} t={DEFAULT_THEME} />
    </Box>,
    {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    }
  )

  instance.unmount()
  instance.cleanup()

  return streams.getOutput()
}

afterEach(() => {
  clearSpawnHistory()
  resetOverlayState()
  resetTurnState()
  turnController.fullReset()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('AgentDockView', () => {
  it('renders nothing without subagents', () => {
    expect(renderView([]).trim()).toBe('')
  })

  it('renders semantic rows with bounded, sanitized activity', () => {
    const output = renderView([
      makeItem({
        goal: 'inspect the layout',
        id: 'a',
        index: 0,
        notes: ['private progress note'],
        startedAt: NOW - 5000,
        tools: ['Read("private/path")']
      }),
      makeItem({ durationSeconds: 12, goal: 'write tests', id: 'b', index: 1, status: 'completed', summary: 'tests pass' }),
      makeItem({ goal: 'verify build', id: 'c', index: 2, status: 'failed' })
    ])

    expect(output).toContain('agents · 1 running · 1 ready · 1 blocked · /agents ↗')
    expect(output).not.toContain('1/3 active')
    expect(output).toContain('● inspect')
    expect(output).toContain('reading files')
    expect(output).toContain('5s')
    expect(output).toContain('✓ write')
    expect(output).toContain('result ready')
    expect(output).toContain('12s')
    expect(output).toContain('✗ verify')
    expect(output).toContain('failed')
    expect(output).not.toContain('tests pass')
    expect(output).not.toContain('private/path')
    expect(output).not.toContain('private progress note')
  })

  it('shows only a one-line summary on narrow terminals', () => {
    const output = renderView(
      [
        makeItem({ goal: 'must not render', id: 'a', index: 0, startedAt: NOW - 5000 }),
        makeItem({ goal: 'also hidden', id: 'b', index: 1, status: 'completed' })
      ],
      50
    )

    expect(output).toContain('agents · /agents ↗ · 1/2 active · 5s')
    expect(output).not.toContain('must not render')
    expect(output).not.toContain('also hidden')
  })

  it('preserves the drill-down cue before truncating a very narrow summary', () => {
    const output = renderView([makeItem({ id: 'a', index: 0, startedAt: NOW - 5000 })], 32)

    expect(output).toContain('agents · /agents ↗')
    expect(output.trim().split('\n')).toHaveLength(1)
  })

  it('caps rows at three and reports overflow activity', () => {
    const names = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf']

    const output = renderView(
      Array.from({ length: 7 }, (_, index) =>
        makeItem({ goal: names[index], id: `agent-${index}`, index, status: index === 6 ? 'running' : 'completed' })
      )
    )

    expect(output).toContain('… 4 more · 4 done')
    expect(output).not.toContain('+4 more')
    expect(output).toContain('agent 1')
    expect(output).toContain('agent 2')
    expect(output).toContain('agent 7')
    expect(output).not.toContain('agent 3')
    expect(output).not.toContain('agent 4')
  })

  it('renders a positive active-overflow count', () => {
    const output = renderView(Array.from({ length: 6 }, (_, index) => makeItem({ id: `live-${index}`, index })))

    expect(output).toContain('… 3 more · 3 running')
    expect(output).not.toContain('3 active')
  })

  it('uses truthful terminal counts instead of zero-running language', () => {
    const output = renderView([
      makeItem({ id: 'done-a', index: 0, status: 'completed', summary: 'ready' }),
      makeItem({ id: 'done-b', index: 1, status: 'completed' }),
      makeItem({ id: 'done-c', index: 2, status: 'completed', summary: 'ready' }),
      makeItem({ id: 'blocked', index: 3, status: 'timeout' })
    ])

    expect(output).toContain('agents · 3 done · 1 blocked · /agents ↗')
    expect(output).not.toContain('0 running')
  })

  it.each([60, 80, 86])('keeps every framed line exactly %i terminal cells wide', cols => {
    const output = renderView(
      [
        makeItem({ id: 'inspect', index: 0, startedAt: NOW - 5000, tools: ['Read("private/path")'] }),
        makeItem({ id: 'review', index: 1, status: 'timeout' }),
        makeItem({ id: 'regression', index: 2, status: 'completed', summary: 'ready' }),
        makeItem({ id: 'overflow', index: 3, status: 'queued' })
      ],
      cols
    )

    const lines = output.split('\n').filter(line => /^[╭│╰]/.test(line))

    expect(lines).toHaveLength(6)
    expect(lines[0]).toMatch(/^╭.*╮$/)
    expect(lines.at(-1)).toMatch(/^╰.*╯$/)
    expect(lines.every(line => stringWidth(line) === cols)).toBe(true)
  })

  it('marks omitted header counts explicitly at the exact 60-column frame floor', () => {
    const output = renderView(
      [
        makeItem({ id: 'run', index: 0, status: 'running' }),
        makeItem({ id: 'queue', index: 1, status: 'queued' }),
        makeItem({ id: 'ready', index: 2, status: 'completed', summary: 'ready' }),
        makeItem({ id: 'blocked', index: 3, status: 'failed' })
      ],
      60
    )

    const header = output.split('\n').find(line => line.startsWith('╭')) ?? ''

    expect(header).toContain(' · … · /agents ↗')
    expect(stringWidth(header)).toBe(60)
  })

  it('opens the existing overlay route on click without bubbling', () => {
    const onOpen = vi.fn()
    const stopImmediatePropagation = vi.fn()

    const element = AgentDockView({
      cols: 100,
      nowMs: NOW,
      onOpen,
      subagents: [makeItem({ id: 'a', index: 0 })],
      t: DEFAULT_THEME
    }) as React.ReactElement<{ onClick: (event: { stopImmediatePropagation: () => void }) => void }>

    element.props.onClick({ stopImmediatePropagation })

    expect(stopImmediatePropagation).toHaveBeenCalledOnce()
    expect(onOpen).toHaveBeenCalledOnce()
  })

  it('flows gateway lifecycle through the live dock, overlay route, and history archive', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
    const appended: Msg[] = []
    const handler = createGatewayEventHandler(buildFlowCtx(appended))
    const openAgents = () => patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'inspect runtime flow', subagent_id: 'flow-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'flow-agent', status: 'running' }])
    let output = renderView(getTurnState().subagents, 80)
    expect(output).toContain('agents · 1 running · /agents ↗')
    expect(output).toContain('● inspect')

    handler({
      payload: {
        goal: 'inspect runtime flow',
        subagent_id: 'flow-agent',
        task_index: 0,
        tool_name: 'Read',
        tool_preview: 'private/runtime/path'
      },
      type: 'subagent.tool'
    } as any)
    handler({
      payload: { goal: 'inspect runtime flow', subagent_id: 'flow-agent', task_index: 0, text: 'private progress note' },
      type: 'subagent.progress'
    } as any)

    output = renderView(getTurnState().subagents, 80)
    expect(output).toContain('reading files')
    expect(output).not.toContain('private/runtime/path')
    expect(output).not.toContain('private progress note')

    const clickable = AgentDockView({
      cols: 80,
      nowMs: NOW,
      onOpen: openAgents,
      subagents: getTurnState().subagents,
      t: DEFAULT_THEME
    }) as React.ReactElement<{ onClick: (event: { stopImmediatePropagation: () => void }) => void }>

    clickable.props.onClick({ stopImmediatePropagation: vi.fn() })
    expect(getOverlayState()).toMatchObject({ agents: true, agentsInitialHistoryIndex: 0 })

    handler({
      payload: {
        duration_seconds: 9,
        goal: 'inspect runtime flow',
        status: 'completed',
        subagent_id: 'flow-agent',
        summary: 'private completion summary',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)
    handler({
      payload: { goal: 'inspect runtime flow', subagent_id: 'flow-agent', task_index: 0, text: 'late private note' },
      type: 'subagent.progress'
    } as any)

    expect(getTurnState().subagents).toMatchObject([
      { durationSeconds: 9, id: 'flow-agent', status: 'completed', summary: 'private completion summary' }
    ])
    expect(projectAgentDock(getTurnState().subagents, { nowMs: NOW, width: 80 }).hidden).toBe(false)
    expect(getSpawnHistory()).toEqual([])

    handler({ payload: { text: 'parent synthesis complete' }, type: 'message.complete' } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { durationSeconds: 9, id: 'flow-agent', status: 'completed', summary: 'private completion summary' }
    ])
    expect(appended).toContainEqual(expect.objectContaining({ role: 'assistant', text: 'parent synthesis complete' }))

    const overlayStreams = makeStreams(100)
    Object.assign(overlayStreams.stdin, {
      isTTY: true,
      ref: vi.fn(),
      setRawMode: vi.fn(),
      unref: vi.fn()
    })

    const overlay = renderSync(
      <AgentsOverlay
        gw={{ request: vi.fn(async () => ({})) } as any}
        initialHistoryIndex={0}
        onClose={() => patchOverlayState({ agents: false })}
        t={DEFAULT_THEME}
      />,
      {
        patchConsole: false,
        stderr: overlayStreams.stderr as unknown as NodeJS.WriteStream,
        stdin: overlayStreams.stdin as unknown as NodeJS.ReadStream,
        stdout: overlayStreams.stdout as unknown as NodeJS.WriteStream
      }
    )

    await vi.advanceTimersByTimeAsync(0)

    expect(overlayStreams.getOutput()).toContain('Last turn · finished')
    expect(overlayStreams.getOutput()).toContain('inspect runtime flow')
    expect(overlayStreams.getOutput()).toContain('controls locked')
    expect(overlayStreams.getOutput()).not.toContain('No subagents this turn')

    overlay.unmount()
    overlay.cleanup()
  })

  it('preserves background subagents across parent turns and archives terminal wake-up results', () => {
    const appended: Msg[] = []
    const handler = createGatewayEventHandler(buildFlowCtx(appended))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'inspect background flow', subagent_id: 'background-agent', task_index: 0 },
      type: 'subagent.spawn_requested'
    } as any)
    handler({ payload: { text: 'delegation dispatched' }, type: 'message.complete' } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'background-agent', status: 'queued' }])
    expect(getSpawnHistory()).toEqual([])

    handler({ payload: {}, type: 'message.start' } as any)
    expect(getTurnState().subagents).toMatchObject([{ id: 'background-agent', status: 'queued' }])

    handler({
      payload: { goal: 'inspect background flow', subagent_id: 'background-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)
    expect(getTurnState().subagents).toMatchObject([{ id: 'background-agent', status: 'running' }])

    handler({
      payload: {
        duration_seconds: 12,
        goal: 'inspect background flow',
        status: 'completed',
        subagent_id: 'background-agent',
        summary: 'verified background result',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'background-agent', status: 'completed' }])
    expect(getSpawnHistory()).toEqual([])
    handler({ payload: { text: 'background result synthesized' }, type: 'message.complete' } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { durationSeconds: 12, id: 'background-agent', status: 'completed', summary: 'verified background result' }
    ])
  })

  it('preserves a detached child across gateway recovery until its terminal event archives it', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'survive gateway recovery', subagent_id: 'recovery-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)
    handler({ payload: { text: 'delegation dispatched' }, type: 'message.complete' } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'recovery-agent', status: 'running' }])
    expect(getSpawnHistory()).toEqual([])

    turnController.reset()

    expect(getTurnState().subagents).toMatchObject([{ id: 'recovery-agent', status: 'running' }])
    expect(getSpawnHistory()).toEqual([])

    handler({
      payload: {
        duration_seconds: 17,
        goal: 'survive gateway recovery',
        status: 'completed',
        subagent_id: 'recovery-agent',
        summary: 'completed after gateway recovery',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { durationSeconds: 17, id: 'recovery-agent', status: 'completed', summary: 'completed after gateway recovery' }
    ])
  })

  it('keeps sequential delegations in one live tree and archives one snapshot at parent turn end', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'inspect first', subagent_id: 'sequential-a', task_index: 0 },
      type: 'subagent.start'
    } as any)
    handler({
      payload: { goal: 'inspect first', status: 'completed', subagent_id: 'sequential-a', task_index: 0 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'sequential-a', status: 'completed' }])
    expect(getSpawnHistory()).toEqual([])

    handler({
      payload: { goal: 'review second', subagent_id: 'sequential-b', task_index: 1 },
      type: 'subagent.start'
    } as any)
    handler({
      payload: { goal: 'review second', status: 'completed', subagent_id: 'sequential-b', task_index: 1 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toMatchObject([
      { id: 'sequential-a', status: 'completed' },
      { id: 'sequential-b', status: 'completed' }
    ])
    expect(getSpawnHistory()).toEqual([])

    handler({ payload: { text: 'sequential delegation complete' }, type: 'message.complete' } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { id: 'sequential-a', status: 'completed' },
      { id: 'sequential-b', status: 'completed' }
    ])
  })

  it('archives a detached completion that lands after submit but before message.start', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'finish before next start', subagent_id: 'prestart-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)
    handler({ payload: { text: 'delegation dispatched' }, type: 'message.complete' } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'prestart-agent', status: 'running' }])
    expect(getSpawnHistory()).toEqual([])

    markSubmitting()
    handler({
      payload: {
        duration_seconds: 7,
        goal: 'finish before next start',
        status: 'completed',
        subagent_id: 'prestart-agent',
        summary: 'completed during submit round-trip',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'prestart-agent', status: 'completed' }])
    expect(getSpawnHistory()).toEqual([])

    handler({ payload: {}, type: 'message.start' } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { durationSeconds: 7, id: 'prestart-agent', status: 'completed', summary: 'completed during submit round-trip' }
    ])
  })

  it('archives a detached child immediately when it completes after a real parent turn ends', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'finish while parent idle', subagent_id: 'idle-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)
    handler({ payload: { text: 'delegation dispatched' }, type: 'message.complete' } as any)
    handler({
      payload: {
        duration_seconds: 13,
        goal: 'finish while parent idle',
        status: 'completed',
        subagent_id: 'idle-agent',
        summary: 'completed while parent idle',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { durationSeconds: 13, id: 'idle-agent', status: 'completed', summary: 'completed while parent idle' }
    ])
  })

  it('keeps terminal tombstones across recovery reset and clears them at fullReset', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    const event = {
      payload: { goal: 'reject stale replay', subagent_id: 'tombstone-agent', task_index: 0 },
      type: 'subagent.start'
    } as any

    handler(event)
    handler({
      payload: { goal: 'reject stale replay', status: 'completed', subagent_id: 'tombstone-agent', task_index: 0 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)

    turnController.reset()
    handler(event)
    expect(getTurnState().subagents).toEqual([])

    turnController.fullReset()
    handler(event)
    expect(getTurnState().subagents).toMatchObject([{ id: 'tombstone-agent', status: 'running' }])
  })

  it('allows a later turn to reuse the same fallback subagent identity', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    const start = {
      payload: { goal: 'repeat fallback delegation', task_index: 0 },
      type: 'subagent.start'
    } as any

    handler({ payload: {}, type: 'message.start' } as any)
    handler(start)
    handler({
      payload: { goal: 'repeat fallback delegation', status: 'completed', task_index: 0 },
      type: 'subagent.complete'
    } as any)
    handler({ payload: { text: 'first turn done' }, type: 'message.complete' } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)

    handler(start)
    expect(getTurnState().subagents).toEqual([])

    markSubmitting()
    handler({ payload: {}, type: 'message.start' } as any)
    handler(start)

    expect(getTurnState().subagents).toMatchObject([
      { id: 'sa:0:repeat fallback delegation', status: 'running' }
    ])
    expect(getSpawnHistory()).toHaveLength(1)
  })

  it('archives an all-terminal outgoing tree when fullReset abandons its active turn', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'archive outgoing session', subagent_id: 'outgoing-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)
    handler({
      payload: { goal: 'archive outgoing session', status: 'completed', subagent_id: 'outgoing-agent', task_index: 0 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'outgoing-agent', status: 'completed' }])
    expect(getSpawnHistory()).toEqual([])

    turnController.fullReset()

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([{ id: 'outgoing-agent', status: 'completed' }])
  })

  it('makes bare idle preserve pending background work and archive it once terminal', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'survive visible history reset', subagent_id: 'visible-reset-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)
    handler({ payload: { text: 'delegation dispatched' }, type: 'message.complete' } as any)

    turnController.idle()

    expect(getTurnState().subagents).toMatchObject([{ id: 'visible-reset-agent', status: 'running' }])
    expect(getSpawnHistory()).toEqual([])

    markSubmitting()
    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: {
        goal: 'survive visible history reset',
        status: 'completed',
        subagent_id: 'visible-reset-agent',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'visible-reset-agent', status: 'completed' }])

    turnController.idle()

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([{ id: 'visible-reset-agent', status: 'completed' }])
  })

  it('waits for every child to become terminal before archiving the live tree', () => {
    const handler = createGatewayEventHandler(buildFlowCtx([]))

    handler({ payload: { goal: 'inspect first', subagent_id: 'first-agent', task_index: 0 }, type: 'subagent.start' } as any)
    handler({ payload: { goal: 'review second', subagent_id: 'second-agent', task_index: 1 }, type: 'subagent.start' } as any)
    handler({
      payload: { goal: 'inspect first', status: 'completed', subagent_id: 'first-agent', task_index: 0 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toMatchObject([
      { id: 'first-agent', status: 'completed' },
      { id: 'second-agent', status: 'running' }
    ])
    expect(getSpawnHistory()).toEqual([])

    handler({
      payload: { goal: 'review second', status: 'failed', subagent_id: 'second-agent', task_index: 1 },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { id: 'first-agent', status: 'completed' },
      { id: 'second-agent', status: 'failed' }
    ])
  })

  it('keeps a background tree visible when its wake-up turn is interrupted', () => {
    const appended: Msg[] = []
    const handler = createGatewayEventHandler(buildFlowCtx(appended))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'survive parent interrupt', subagent_id: 'interrupt-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)
    handler({ payload: { text: 'delegation dispatched' }, type: 'message.complete' } as any)
    handler({ payload: {}, type: 'message.start' } as any)

    turnController.interruptTurn({
      appendMessage: (msg: Msg) => appended.push(msg),
      gw: { request: vi.fn(async () => ({})) } as any,
      sid: 'session-1',
      sys: vi.fn()
    })

    expect(getTurnState().subagents).toMatchObject([{ id: 'interrupt-agent', status: 'running' }])
    expect(getSpawnHistory()).toEqual([])
  })

  it('preserves a running child when its spawn turn is interrupted before message.complete', () => {
    const appended: Msg[] = []
    const handler = createGatewayEventHandler(buildFlowCtx(appended))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'survive spawn interrupt', subagent_id: 'spawn-interrupt-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)

    turnController.interruptTurn({
      appendMessage: (msg: Msg) => appended.push(msg),
      gw: { request: vi.fn(async () => ({})) } as any,
      sid: 'session-1',
      sys: vi.fn()
    })

    expect(getTurnState().subagents).toMatchObject([{ id: 'spawn-interrupt-agent', status: 'running' }])

    handler({
      payload: {
        duration_seconds: 8,
        goal: 'survive spawn interrupt',
        status: 'completed',
        subagent_id: 'spawn-interrupt-agent',
        summary: 'completed after interrupt',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { durationSeconds: 8, id: 'spawn-interrupt-agent', status: 'completed' }
    ])
  })

  it('preserves a running child when its spawn turn errors before message.complete', () => {
    const appended: Msg[] = []
    const handler = createGatewayEventHandler(buildFlowCtx(appended))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'survive spawn error', subagent_id: 'spawn-error-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)

    turnController.recordError()

    expect(getTurnState().subagents).toMatchObject([{ id: 'spawn-error-agent', status: 'running' }])

    handler({
      payload: {
        duration_seconds: 9,
        goal: 'survive spawn error',
        status: 'failed',
        subagent_id: 'spawn-error-agent',
        summary: 'failed after parent error',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { durationSeconds: 9, id: 'spawn-error-agent', status: 'failed' }
    ])
  })

  it('preserves a late background start across the next turn and archives it at wake-up completion', () => {
    const appended: Msg[] = []
    const handler = createGatewayEventHandler(buildFlowCtx(appended))

    handler({ payload: {}, type: 'message.start' } as any)
    handler({ payload: { text: 'delegation dispatched' }, type: 'message.complete' } as any)
    handler({
      payload: { goal: 'inspect late lifecycle', subagent_id: 'late-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'late-agent', status: 'running' }])

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'inspect late lifecycle', subagent_id: 'late-agent', task_index: 0, text: 'still running' },
      type: 'subagent.progress'
    } as any)

    expect(getTurnState().subagents).toMatchObject([{ id: 'late-agent', status: 'running' }])

    handler({
      payload: {
        duration_seconds: 11,
        goal: 'inspect late lifecycle',
        status: 'completed',
        subagent_id: 'late-agent',
        summary: 'late result',
        task_index: 0
      },
      type: 'subagent.complete'
    } as any)

    expect(getTurnState().subagents).toMatchObject([
      { durationSeconds: 11, id: 'late-agent', status: 'completed', summary: 'late result' }
    ])
    expect(getSpawnHistory()).toEqual([])

    handler({ payload: { text: 'late result synthesized' }, type: 'message.complete' } as any)

    expect(getTurnState().subagents).toEqual([])
    expect(getSpawnHistory()).toHaveLength(1)
    expect(getSpawnHistory()[0]?.subagents).toMatchObject([
      { durationSeconds: 11, id: 'late-agent', status: 'completed', summary: 'late result' }
    ])

    handler({ payload: {}, type: 'message.start' } as any)
    handler({
      payload: { goal: 'review new tree', subagent_id: 'new-agent', task_index: 0 },
      type: 'subagent.start'
    } as any)

    expect(getTurnState().subagents.map(item => item.id)).toEqual(['new-agent'])
    expect(getSpawnHistory()).toHaveLength(1)
  })

})

describe('AgentDock live wrapper', () => {
  it('reacts to height-only resizes and restores the one-line short-terminal summary safely', async () => {
    patchTurnState({
      subagents: [
        makeItem({ id: 'run', index: 0, status: 'running' }),
        makeItem({ id: 'queue', index: 1, status: 'queued' }),
        makeItem({ id: 'done', index: 2, status: 'completed' })
      ]
    })

    const streams = makeStreams(80, 14)
    const previousRows = process.stdout.rows
    const resizeListenersBefore = process.stdout.listenerCount('resize')
    let instance: ReturnType<typeof renderSync> | null = null

    try {
      process.stdout.rows = 14

      instance = renderSync(<AgentDock cols={80} onOpen={() => {}} t={DEFAULT_THEME} />, {
        patchConsole: false,
        stderr: streams.stderr as unknown as NodeJS.WriteStream,
        stdin: streams.stdin as unknown as NodeJS.ReadStream,
        stdout: streams.stdout as unknown as NodeJS.WriteStream
      })

      await vi.waitFor(() => expect(streams.getOutput()).toContain('agents · /agents ↗ · 2/3 active'))
      expect(streams.getOutput()).not.toContain('╭')

      process.stdout.rows = 16
      process.stdout.emit('resize')

      await vi.waitFor(() => expect(streams.getOutput()).toContain('╭'))
      const outputBeforeReturnToShort = streams.getOutput().length

      process.stdout.rows = 15
      process.stdout.emit('resize')

      await vi.waitFor(() =>
        expect(streams.getOutput().slice(outputBeforeReturnToShort)).toContain('agents · /agents ↗ · 2/3 active')
      )
    } finally {
      instance?.unmount()
      instance?.cleanup()
      process.stdout.rows = previousRows
    }

    expect(process.stdout.listenerCount('resize')).toBe(resizeListenersBefore)
  })

  it('subscribes without mutation, advances elapsed time, and stops its clock when work finishes', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
    const setIntervalSpy = vi.spyOn(globalThis, 'setInterval')
    const clearIntervalSpy = vi.spyOn(globalThis, 'clearInterval')
    const subagents = [makeItem({ id: 'a', index: 0, startedAt: NOW - 5000 })]
    patchTurnState({ subagents })
    const before = getTurnState()
    const streams = makeStreams()

    const instance = renderSync(<AgentDock cols={100} onOpen={() => {}} t={DEFAULT_THEME} />, {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    await vi.advanceTimersByTimeAsync(0)

    expect(getTurnState()).toBe(before)
    expect(getTurnState().subagents).toBe(subagents)
    expect(streams.getOutput()).toContain('5s')
    const dockTimerIndex = setIntervalSpy.mock.calls.findIndex(call => call[1] === 1000)
    expect(dockTimerIndex).toBeGreaterThanOrEqual(0)
    const dockTimer = setIntervalSpy.mock.results[dockTimerIndex]!.value

    await vi.advanceTimersByTimeAsync(1000)
    expect(streams.getOutput()).toContain('6s')

    patchTurnState({
      subagents: [makeItem({ durationSeconds: 6, id: 'a', index: 0, startedAt: NOW - 5000, status: 'completed' })]
    })
    await vi.advanceTimersByTimeAsync(0)
    expect(clearIntervalSpy).toHaveBeenCalledWith(dockTimer)

    instance.unmount()
    instance.cleanup()
  })
})
