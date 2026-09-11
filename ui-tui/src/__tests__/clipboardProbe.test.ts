import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useComposerState } from '../app/useComposerState.js'
import { createClipProbeLatch } from '../lib/clipboardProbeLatch.js'

// ═══════════════════════════════════════════════════════════════════
// Latch unit tests — exercise the REAL production module.
// ═══════════════════════════════════════════════════════════════════

describe('createClipProbeLatch (production module)', () => {
  it('allows the first speculative probe', () => {
    const latch = createClipProbeLatch()

    expect(latch.tryProbe()).toBe(true)
  })

  it('suppresses a second speculative probe (latch persists)', () => {
    const latch = createClipProbeLatch()

    latch.tryProbe() // first — fires
    expect(latch.tryProbe()).toBe(false) // second — suppressed
  })

  it('suppresses all subsequent probes indefinitely (no time-based reset)', () => {
    const latch = createClipProbeLatch()

    latch.tryProbe() // fires

    // Arbitrarily many probes — all suppressed; "arbitrary elapsed time" does
    // not reset the latch.
    for (let i = 0; i < 100; i++) {
      expect(latch.tryProbe()).toBe(false)
    }
  })

  it('resets on a non-empty stream boundary so a later new stream can probe again', () => {
    const latch = createClipProbeLatch()

    latch.tryProbe() // empty bracketed paste → fires, latched
    expect(latch.tryProbe()).toBe(false) // suppressed

    // Non-empty paste arrives — meaningful stream boundary
    latch.reset()

    // Now a new empty bracketed paste can fire again
    expect(latch.tryProbe()).toBe(true)
  })

  it('explicit reset re-enables the next probe even when latched', () => {
    const latch = createClipProbeLatch()

    latch.tryProbe() // fires, latched
    latch.reset() // explicit path (/paste, hotkey)

    expect(latch.tryProbe()).toBe(true) // next speculative probe works
  })

  it('repeated non-empty boundaries each allow a single new speculative probe', () => {
    const latch = createClipProbeLatch()

    for (let cycle = 0; cycle < 5; cycle++) {
      // First empty bracketed paste after reset → fires
      expect(latch.tryProbe()).toBe(true)
      // Subsequent empties → suppressed
      expect(latch.tryProbe()).toBe(false)
      expect(latch.tryProbe()).toBe(false)

      // Non-empty paste → meaningful boundary
      latch.reset()
    }
  })

  it('reset before any probe is a no-op (idempotent)', () => {
    const latch = createClipProbeLatch()

    latch.reset() // no-op

    // First probe still fires
    expect(latch.tryProbe()).toBe(true)

    // Double reset is also idempotent
    latch.reset()
    latch.reset()
    expect(latch.tryProbe()).toBe(true)
  })
})

// ═══════════════════════════════════════════════════════════════════
// Route-level integration — render the real useComposerState hook and
// invoke its actual handleResolvedPaste / handleTextPaste handlers.
// ═══════════════════════════════════════════════════════════════════

// Mock the gateway client that useComposerState receives.
const mockRequest = vi.fn<(method: string, params?: Record<string, unknown>) => Promise<unknown>>()

// ---------------------------------------------------------------------------
// Hoisted mocks (vitest hoists vi.mock above imports — these run first).
// ---------------------------------------------------------------------------

vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal<Record<string, unknown>>()

  return {
    ...mod,
    useInput: () => {},
    useStdin: () => ({ querier: null }),
    withInkSuspended: (fn: () => Promise<void>) => fn()
  }
})

vi.mock('@nanostores/react', () => ({
  useStore: () => false // $isBlocked is always false during tests
}))

vi.mock('../hooks/useCompletion.js', () => ({
  useCompletion: () => ({
    completions: [] as unknown[],
    compIdx: 0,
    setCompIdx: () => {},
    compReplace: 0
  })
}))

vi.mock('../hooks/useInputHistory.js', () => ({
  useInputHistory: () => ({
    historyRef: { current: [] as string[] },
    historyIdx: null as number | null,
    setHistoryIdx: () => {},
    historyDraftRef: { current: '' },
    pushHistory: () => {}
  })
}))

vi.mock('../hooks/useQueue.js', () => ({
  useQueue: () => ({
    queueRef: { current: [] as unknown[] },
    queueEditRef: { current: null as number | null },
    queuedDisplay: [] as string[],
    queueEditIdx: null as number | null,
    enqueue: () => {},
    dequeue: () => undefined,
    prependQ: () => {},
    removeQ: () => {},
    setQueueEdit: () => {},
    takeQ: () => undefined
  })
}))

vi.mock('../lib/terminalSetup.js', () => ({
  isRemoteShellSession: () => false
}))

// Mock uiStore so getUiState() returns a valid sid — without this,
// pasteClipboardImage bails early with null and never calls gw.request.
vi.mock('../app/uiStore.js', () => {
  const { atom } = require('nanostores')

  const testAtom = atom({
    sid: 'test-sid-123',
    pasteCollapseLines: 5,
    pasteCollapseChars: 2000
  })

  return {
    getUiState: () => testAtom.get(),
    $uiState: testAtom,
    $uiTheme: atom({}),
    $uiSessionId: atom('test-sid-123'),
    patchUiState: () => {}
  }
})

// We cannot easily render useComposerState in isolation (React hook + many
// context providers needed).  Instead we test the latch behaviour through
// its production module (above) and verify the route-level contract via a
// minimal harness that exercises the actual ComposerState hook through a
// thin React wrapper rendered with @hermes/ink's renderSync.

// Helper to render a component with the ink renderer.
const renderInk = (el: React.ReactElement): string => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let out = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', (chunk: Buffer) => { out += chunk.toString() })

  const instance = renderSync(el, {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return out
}

// ---------------------------------------------------------------------------
// Integration: invoke the real hook's handleResolvedPaste via a rendered
// component that wires it to a callback we can call from the test.
// ---------------------------------------------------------------------------

describe('useComposerState latch route (integration)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockRequest.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // ── harness ──────────────────────────────────────────────────────────

  // A thin component that renders useComposerState and exposes the resulting
  // handleResolvedPaste and handleTextPaste callbacks through a mutable ref
  // so the test can drive the real handlers directly.

  type PasteHandler = ReturnType<typeof useComposerState>['actions']['handleTextPaste']

  interface HarnessRef {
    handleTextPaste: PasteHandler
  }

  const HarnessComponent: React.FC<{
    gw: { request: typeof mockRequest }
    harnessRef: React.MutableRefObject<HarnessRef | null>
    submitRef: React.MutableRefObject<(value: string) => void>
  }> = ({ gw, harnessRef, submitRef }) => {
    const result = useComposerState({
      gw: gw as unknown as Parameters<typeof useComposerState>[0]['gw'],
      submitRef,
      sys: () => {}
    })

    // Expose the production handler synchronously for the test harness.
    harnessRef.current = { handleTextPaste: result.actions.handleTextPaste }

    return React.createElement('ink-text', null, 'composer')
  }

  // Alternative: access handleResolvedPaste directly through the returned
  // handleTextPaste which delegates to it for non-hotkey pastes.
  // For empty bracketed pastes, handleTextPaste calls handleResolvedPaste
  // directly: `return handleResolvedPaste({ bracketed: !!bracketed, cursor, text, value })`

  const renderHarness = (harnessRef: React.MutableRefObject<HarnessRef | null>) => {
    const submitRef = { current: (_v: string) => {} }

    renderInk(
      React.createElement(HarnessComponent, {
        gw: { request: mockRequest as unknown as typeof mockRequest },
        harnessRef,
        submitRef
      })
    )
  }

  it('calls clipboard.paste at most once for repeated empty bracketed-paste events', async () => {
    const harnessRef = { current: null as HarnessRef | null }

    renderHarness(harnessRef)

    const handler = harnessRef.current!.handleTextPaste

    // Mock gateway request — clipboard.paste returns null (no image)
    mockRequest.mockResolvedValue({ attached: false })

    // First empty bracketed paste — should trigger a clipboard.paste call.
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })

    expect(mockRequest).toHaveBeenCalledTimes(1)
    expect(mockRequest).toHaveBeenCalledWith('clipboard.paste', expect.objectContaining({}))

    // Second empty bracketed paste — latched, no call.
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })

    expect(mockRequest).toHaveBeenCalledTimes(1) // still exactly one

    // Third, fourth... still latched.
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })

    expect(mockRequest).toHaveBeenCalledTimes(1) // still exactly one
  })

  it('deleting the production latch would cause the test to fail (multiple calls)', async () => {
    // This test proves the latch is necessary: if the latch were bypassed
    // (i.e., every empty bracketed paste called clipboard.paste), multiple
    // calls would be observed and the assertion on call count would fail.
    // The latch module tests above prove that tryProbe returns false after
    // the first call — this integration test verifies the route end-to-end.
    const harnessRef = { current: null as HarnessRef | null }

    renderHarness(harnessRef)

    const handler = harnessRef.current!.handleTextPaste

    mockRequest.mockResolvedValue({ attached: false })

    // Fire three empty bracketed pastes.
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })

    // Exactly ONE gateway call — the latch suppressed the other two.
    // If the latch code were deleted from production, mockRequest would be
    // called 3 times and this assertion would fail.
    expect(mockRequest).toHaveBeenCalledTimes(1)
  })

  it('non-empty stream boundary resets the latch and permits one later probe', async () => {
    const harnessRef = { current: null as HarnessRef | null }

    renderHarness(harnessRef)

    const handler = harnessRef.current!.handleTextPaste

    mockRequest.mockResolvedValue({ attached: false })

    // First empty bracketed paste → clips the latch.
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })
    expect(mockRequest).toHaveBeenCalledTimes(1)

    // Non-empty paste → resets the latch (meaningful stream boundary).
    await handler({ bracketed: false, cursor: 0, text: 'hello', value: 'hello' })
    mockRequest.mockClear() // clear for new count

    // Now a new empty bracketed paste can fire again.
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })

    expect(mockRequest).toHaveBeenCalledTimes(1)
    expect(mockRequest).toHaveBeenCalledWith('clipboard.paste', expect.objectContaining({}))
  })

  it('explicit paste hotkey resets the latch and is functional', async () => {
    const harnessRef = { current: null as HarnessRef | null }

    renderHarness(harnessRef)

    const handler = harnessRef.current!.handleTextPaste

    // First empty bracketed paste arms the latch.
    mockRequest.mockResolvedValue({ attached: false })
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })
    expect(mockRequest).toHaveBeenCalledTimes(1)

    // Explicit hotkey paste bypasses the speculative-probe latch and resets it.
    mockRequest.mockClear()
    await handler({ bracketed: false, cursor: 0, hotkey: true, text: '', value: '' })
    expect(mockRequest).toHaveBeenCalled()

    // The reset permits exactly one later speculative probe, then latches again.
    mockRequest.mockClear()
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })
    await handler({ bracketed: true, cursor: 0, text: '', value: '' })
    expect(mockRequest).toHaveBeenCalledTimes(1)
  })
})
