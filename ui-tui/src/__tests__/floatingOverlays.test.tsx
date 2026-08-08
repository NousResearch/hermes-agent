import { PassThrough } from 'node:stream'

import { Box, render, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import { $isBlocked, type OverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { patchUiState } from '../app/uiStore.js'
import { FloatingOverlays } from '../components/appOverlays.js'
import { AmbientDock, openWidget } from '../sdk/host.js'
import { defineWidgetApp, removeWidgetApp } from '../sdk/registry.js'

// ── Stubs ─────────────────────────────────────────────────────────────
// The real panel widgets mount their own input handlers + gateway RPC
// calls. For a layout regression we only care that FloatingOverlays puts
// them on screen, so swap each for a single marker row. vi.mock is hoisted
// before the component import above resolves.

vi.mock('../components/activeSessionSwitcher.js', () => ({
  ActiveSessionSwitcher: () => <Text>SESSIONS_PANEL_MARKER</Text>
}))

vi.mock('../components/modelPicker.js', () => ({
  ModelPicker: () => <Text>MODEL_PICKER_MARKER</Text>,
  // Exported helper kept by the real module; not exercised here but kept so
  // the mock still satisfies any sibling import.
  providerIndexAfterClearingFilter: () => -1
}))

vi.mock('../components/petPicker.js', () => ({
  PetPicker: () => <Text>PET_PICKER_MARKER</Text>
}))

vi.mock('../components/skillsHub.js', () => ({
  SkillsHub: () => <Text>SKILLS_HUB_MARKER</Text>
}))

vi.mock('../components/pluginsHub.js', () => ({
  PluginsHub: () => <Text>PLUGINS_HUB_MARKER</Text>
}))

// A non-TTY writable stream that records every frame Ink writes, so the test
// can assert what actually lands on screen. Mirrors how the real renderer
// writes to process.stdout, minus terminal control.
const makeCapturingStdout = () => {
  const stream = new PassThrough()
  Object.defineProperty(stream, 'columns', { configurable: true, value: 100 })
  Object.defineProperty(stream, 'rows', { configurable: true, value: 40 })
  Object.defineProperty(stream, 'isTTY', { configurable: true, value: false })

  let captured = ''
  stream.on('data', (chunk: Buffer) => {
    captured += chunk.toString()
  })

  return {
    stdout: stream as unknown as NodeJS.WriteStream,
    read: () => captured
  }
}

// FloatingOverlays reads $overlayState / $uiTheme / $uiSessionId (nanostores)
// and the gateway context. Seed the stores, wrap in a provider, render, give
// the throttled/microtask-deferred frame a tick to flush, then read the frame.
const renderOverlays = async (overlay: Partial<OverlayState>, completions: { display: string; text: string }[] = []) => {
  resetOverlayState()
  patchOverlayState(overlay)
  patchUiState({ sid: 'sess-test' })

  const { stdout, read } = makeCapturingStdout()

  const noop = vi.fn()

  const instance = await render(
    <GatewayProvider value={{ gw: {} as never, rpc: noop as never }}>
      <FloatingOverlays
        cols={100}
        compIdx={0}
        completions={completions}
        onActiveSessionClose={noop}
        onActiveSessionSelect={noop}
        onModelSelect={noop}
        onNewLiveSession={noop}
        onNewPromptSession={noop}
        onResumeSelect={noop}
        pagerPageSize={20}
      />
    </GatewayProvider>,
    { stderr: process.stderr, stdin: process.stdin, stdout }
  )

  await new Promise(resolve => setTimeout(resolve, 60))
  instance.unmount()
  await new Promise(resolve => setTimeout(resolve, 20))

  return read()
}

// Minimal slice of ComposerPane's composer box: the prompt is gated on
// $isBlocked exactly like appLayout.tsx, and FloatingOverlays sits above it
// in the same relative parent. This is the composition #69592 broke.
//
// `headroom` reserves blank rows ABOVE the relative composer box so an
// absolute completion (positioned bottom:100% → floats above the box) has
// somewhere to land inside the captured viewport instead of being clipped
// off the top of a non-TTY render.
const ComposerSlice = ({
  completions,
  headroom = 0,
  prompt
}: {
  completions: { display: string; text: string }[]
  headroom?: number
  prompt: string
}) => {
  const isBlocked = React.useSyncExternalStore(
    cb => $isBlocked.subscribe(cb),
    () => $isBlocked.get(),
    () => $isBlocked.get()
  )

  return (
    <Box flexDirection="column">
      {headroom > 0 ? <Box height={headroom} /> : null}
      <Box position="relative">
        <FloatingOverlays
          cols={100}
          compIdx={0}
          completions={completions}
          onActiveSessionClose={vi.fn()}
          onActiveSessionSelect={vi.fn()}
          onModelSelect={vi.fn()}
          onNewLiveSession={vi.fn()}
          onNewPromptSession={vi.fn()}
          onResumeSelect={vi.fn()}
          pagerPageSize={20}
        />

        {!isBlocked && <Text>{prompt}</Text>}
      </Box>
    </Box>
  )
}

const renderSlice = async (
  overlay: Partial<OverlayState>,
  prompt: string,
  completions: { display: string; text: string }[] = [],
  headroom = 0
) => {
  resetOverlayState()
  patchOverlayState(overlay)

  const { stdout, read } = makeCapturingStdout()

  const instance = await render(
    <GatewayProvider value={{ gw: {} as never, rpc: vi.fn() as never }}>
      <ComposerSlice completions={completions} headroom={headroom} prompt={prompt} />
    </GatewayProvider>,
    { stderr: process.stderr, stdin: process.stdin, stdout }
  )

  await new Promise(resolve => setTimeout(resolve, 60))
  instance.unmount()
  await new Promise(resolve => setTimeout(resolve, 20))

  return read()
}

// Real appLayout chrome stack: AmbientDock (top) + relative composer with
// FloatingOverlays + $isBlocked prompt gate + AmbientDock (bottom).
// Review #72208 / issue #69592 required this integrated path — synthetic
// ComposerSlice alone does not mount AmbientDock.
const DOCK_APP_ID = 'test-ambient-dock-69592'
const DOCK_MARKER = 'AMBIENT_DOCK_MARKER'

const AppLayoutChromeSlice = ({
  completions,
  prompt
}: {
  completions: { display: string; text: string }[]
  prompt: string
}) => {
  const isBlocked = React.useSyncExternalStore(
    cb => $isBlocked.subscribe(cb),
    () => $isBlocked.get(),
    () => $isBlocked.get()
  )

  return (
    <Box flexDirection="column">
      <AmbientDock placement="dock-top" />
      <Box position="relative">
        <FloatingOverlays
          cols={100}
          compIdx={0}
          completions={completions}
          onActiveSessionClose={vi.fn()}
          onActiveSessionSelect={vi.fn()}
          onModelSelect={vi.fn()}
          onNewLiveSession={vi.fn()}
          onNewPromptSession={vi.fn()}
          onResumeSelect={vi.fn()}
          pagerPageSize={20}
        />
        {!isBlocked && <Text>{prompt}</Text>}
      </Box>
      <AmbientDock placement="dock-bottom" />
    </Box>
  )
}

const seedDockTopWidget = () => {
  const app = defineWidgetApp({
    help: 'test ambient dock marker',
    id: DOCK_APP_ID,
    init: () => ({}),
    mode: 'ambient' as const,
    reduce: (state: Record<string, never>) => state,
    render: () => <Text>{DOCK_MARKER}</Text>,
    zone: 'dock-top' as const
  })
  openWidget(app, {})
}

const renderAppLayoutChrome = async (
  overlay: Partial<OverlayState>,
  prompt: string,
  completions: { display: string; text: string }[] = []
) => {
  resetOverlayState()
  seedDockTopWidget()
  // openWidget wrote ambient; merge blocking flags without wiping the dock.
  patchOverlayState(overlay)

  const { stdout, read } = makeCapturingStdout()

  const instance = await render(
    <GatewayProvider value={{ gw: {} as never, rpc: vi.fn() as never }}>
      <AppLayoutChromeSlice completions={completions} prompt={prompt} />
    </GatewayProvider>,
    { stderr: process.stderr, stdin: process.stdin, stdout }
  )

  await new Promise(resolve => setTimeout(resolve, 60))
  instance.unmount()
  await new Promise(resolve => setTimeout(resolve, 20))

  return read()
}

describe('FloatingOverlays ambient-dock layout (#69592)', () => {
  beforeEach(() => {
    resetOverlayState()
    patchUiState({ sid: 'sess-test' })
  })

  afterEach(() => {
    resetOverlayState()
    removeWidgetApp(DOCK_APP_ID)
  })

  describe('blocking panels render in-flow (reserve rows, stay visible)', () => {
    it('shows the sessions switcher when sessions is open', async () => {
      const frame = await renderOverlays({ sessions: true })

      // The panel must actually land on screen. With the pre-fix absolute
      // `bottom:100%` wrapper inside the composer's collapsed relative box,
      // this frame was empty — the regression.
      expect(frame).toContain('SESSIONS_PANEL_MARKER')
    })

    it('shows the model picker when modelPicker is open', async () => {
      const frame = await renderOverlays({ modelPicker: true })

      expect(frame).toContain('MODEL_PICKER_MARKER')
    })

    it('shows other blocking hubs the same way', async () => {
      const skills = await renderOverlays({ skillsHub: true })
      expect(skills).toContain('SKILLS_HUB_MARKER')

      const plugins = await renderOverlays({ pluginsHub: true })
      expect(plugins).toContain('PLUGINS_HUB_MARKER')
    })

    it('renders nothing when no overlay and no completions are open', async () => {
      const frame = await renderOverlays({})

      // FloatingOverlays short-circuits to null — no panel or completion
      // content lands in the frame. (The stream still carries Ink's terminal
      // setup escape sequences, which is why we assert on the markers, not on
      // an empty string.)
      expect(frame).not.toContain('SESSIONS_PANEL_MARKER')
      expect(frame).not.toContain('MODEL_PICKER_MARKER')
      expect(frame).not.toContain('reload')
    })
  })

  describe('completions keep the distinct absolute track', () => {
    it('do not block the composer — $isBlocked stays false (the invariant that keeps them absolute)', () => {
      // Completions are deliberately absent from $isBlocked, so the prompt
      // stays live and the parent keeps its height. That is exactly why the
      // completion track can stay absolute (bottom:100% above a live prompt)
      // instead of in-flow like blocking panels.
      resetOverlayState()
      expect($isBlocked.get()).toBe(false)
    })

    it('render above a live prompt without blocking it', async () => {
      // Headroom above the composer box gives the absolute completion
      // (bottom:100% → floats above the box) room to land in the captured
      // viewport. The prompt stays live underneath.
      const frame = await renderSlice(
        {},
        'PROMPT_LIVE_MARKER',
        [{ display: 'reload', text: 'reload' }, { display: 'sessions', text: 'sessions' }],
        20
      )

      expect(frame).toContain('reload')
      expect(frame).toContain('sessions')
      expect(frame).toContain('PROMPT_LIVE_MARKER')
    })
  })

  describe('composer region: picker visible while the prompt is blocked', () => {
    it('hides the prompt and shows the panel when a blocking overlay opens', async () => {
      const frame = await renderSlice({ sessions: true }, 'PROMPT_LIVE_MARKER')

      // The fix's whole point: the panel is visible …
      expect(frame).toContain('SESSIONS_PANEL_MARKER')
      // … while the ordinary prompt is hidden because the composer is blocked.
      expect(frame).not.toContain('PROMPT_LIVE_MARKER')
    })
  })

  describe('integrated AmbientDock + blocking panel (#69592 / #72208 review)', () => {
    it('keeps dock + sessions panel visible while the prompt is blocked', async () => {
      const frame = await renderAppLayoutChrome({ sessions: true }, 'PROMPT_LIVE_MARKER')

      // Real AmbientDock row must paint (this is the chrome that stole height
      // under the old absolute-only overlay path).
      expect(frame).toContain(DOCK_MARKER)
      // Sessions panel stays on screen with the dock occupying rows above.
      expect(frame).toContain('SESSIONS_PANEL_MARKER')
      // Prompt is gated off via $isBlocked, same as appLayout.
      expect(frame).not.toContain('PROMPT_LIVE_MARKER')
    })

    it('keeps dock + model picker visible while the prompt is blocked', async () => {
      const frame = await renderAppLayoutChrome({ modelPicker: true }, 'PROMPT_LIVE_MARKER')

      expect(frame).toContain(DOCK_MARKER)
      expect(frame).toContain('MODEL_PICKER_MARKER')
      expect(frame).not.toContain('PROMPT_LIVE_MARKER')
    })

    it('dock stays while completions leave the prompt live', async () => {
      const frame = await renderAppLayoutChrome({}, 'PROMPT_LIVE_MARKER', [{ display: 'reload', text: 'reload' }])

      expect(frame).toContain(DOCK_MARKER)
      expect(frame).toContain('PROMPT_LIVE_MARKER')
      expect($isBlocked.get()).toBe(false)
    })
  })
})
