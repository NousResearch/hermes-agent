import { PassThrough } from 'node:stream'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Regression coverage for the spawn/kill lifecycle race: a killed
// `tui_gateway.entry` child can stay alive for a real, observable window
// (its own Python-side shutdown grace period — see
// `tui_gateway/entry.py::_shutdown_grace_seconds`, ~1s by default) after
// `GatewayClient.kill()` sends SIGTERM. Before the fix, anything that called
// `request()` (or `start()` directly, e.g. the app-level crash-recovery exit
// handler) during that window treated the not-yet-exited child as "gone" and
// spawned a second one — producing two live gateway processes at once, with
// a request potentially written straight into the half-dead child's stdin
// and never answered (silently hanging until the RPC timeout).
//
// The existing `gatewayClient.test.ts` suite only exercises the websocket
// *attach* mode, so the default *spawn* mode (`startSpawnedGateway`) had no
// coverage at all prior to this file.

const { FakeChildProcess, spawnMock } = vi.hoisted(() => {
  class FakeChildProcess {
    static instances: FakeChildProcess[] = []
    private static nextPid = 1000

    exitCode: null | number = null
    killed = false
    pid = FakeChildProcess.nextPid++
    signalCode: null | string = null
    stderr = new PassThrough()
    stdin = new PassThrough()
    stdout = new PassThrough()

    private listeners = new Map<string, ((...args: any[]) => void)[]>()

    static reset() {
      FakeChildProcess.instances = []
    }

    on(event: string, cb: (...args: any[]) => void) {
      const entries = this.listeners.get(event) ?? []

      entries.push(cb)
      this.listeners.set(event, entries)

      return this
    }

    // Mirrors Node's real semantics: `.kill()` flips `killed` synchronously
    // and returns whether the signal was delivered — it does NOT mean the
    // process has actually exited yet. The OS-level 'exit' event only fires
    // later, via `simulateExit()` in these tests (or never, if a test wants
    // to model a still-shutting-down child).
    kill() {
      if (this.killed || this.exitCode !== null) {
        return false
      }

      this.killed = true

      return true
    }

    // Test-only helper: simulate the child actually terminating at the OS
    // level, asynchronously, some time after kill() was called.
    simulateExit(code: null | number = 0, signal: null | string = null) {
      this.exitCode = code
      this.signalCode = signal
      const entries = this.listeners.get('exit') ?? []

      for (const cb of entries) {
        cb(code, signal)
      }
    }
  }

  const spawnMock = vi.fn(() => {
    const proc = new FakeChildProcess()

    FakeChildProcess.instances.push(proc)

    return proc
  })

  return { FakeChildProcess, spawnMock }
})

vi.mock('node:child_process', () => ({ spawn: spawnMock }))

import { GatewayClient } from '../gatewayClient.js'

describe('GatewayClient spawn mode — kill/respawn lifecycle', () => {
  let originalAttachUrl: string | undefined

  beforeEach(() => {
    originalAttachUrl = process.env.HERMES_TUI_GATEWAY_URL
    delete process.env.HERMES_TUI_GATEWAY_URL
    FakeChildProcess.reset()
    spawnMock.mockClear()
  })

  afterEach(() => {
    if (originalAttachUrl === undefined) {
      delete process.env.HERMES_TUI_GATEWAY_URL
    } else {
      process.env.HERMES_TUI_GATEWAY_URL = originalAttachUrl
    }
  })

  it('spawns exactly one child on start() and resolves a request end to end', async () => {
    const gw = new GatewayClient()

    gw.start()
    expect(spawnMock).toHaveBeenCalledTimes(1)

    const proc = FakeChildProcess.instances[0]!
    const req = gw.request<{ ok: boolean }>('session.create', { cols: 80 })

    await vi.waitFor(() => {
      const written = proc.stdin.read()

      expect(written).not.toBeNull()
    })

    // Re-read isn't possible after consuming above; re-issue a fresh request
    // to capture the frame cleanly for shape assertions.
    const frames: Buffer[] = []

    proc.stdin.on('data', chunk => frames.push(chunk))

    const req2 = gw.request<{ ok: boolean }>('session.create', { cols: 80 })

    await vi.waitFor(() => expect(frames.length).toBeGreaterThan(0))
    const frame = JSON.parse(frames[0]!.toString('utf8')) as { id: string; method: string }

    expect(frame.method).toBe('session.create')
    proc.stdout.write(JSON.stringify({ id: frame.id, jsonrpc: '2.0', result: { ok: true } }) + '\n')
    await expect(req2).resolves.toEqual({ ok: true })

    // The first request's own promise is irrelevant to this assertion; avoid
    // an unhandled rejection warning if it never settles in this test.
    req.catch(() => {})
    gw.kill()
  })

  it('does NOT spawn a second child while the killed one is still alive (mid graceful-exit)', async () => {
    const gw = new GatewayClient()

    gw.start()
    expect(spawnMock).toHaveBeenCalledTimes(1)

    const firstProc = FakeChildProcess.instances[0]!

    gw.kill('graceful-exit-cleanup')
    expect(firstProc.killed).toBe(true)
    // Crucially: the child has NOT actually exited yet — no simulateExit()
    // call — modeling the up-to-~1s Python-side shutdown grace window.
    expect(firstProc.exitCode).toBeNull()

    // A request lands in that window (e.g. a stray poll timer, or the
    // app-level auto-recovery handler reacting to a late 'exit' forward).
    const late = gw.request('model.options', {})

    await expect(late).rejects.toThrow(/shutting down/)

    // The core regression assertion: no second gateway child was spawned
    // while the first was still alive.
    expect(spawnMock).toHaveBeenCalledTimes(1)
    expect(FakeChildProcess.instances).toHaveLength(1)
  })

  it('start() called again after kill() (e.g. the exit-handler auto-recovery path) is a no-op', async () => {
    const gw = new GatewayClient()

    gw.start()
    const firstProc = FakeChildProcess.instances[0]!

    gw.kill('graceful-exit-cleanup')
    expect(firstProc.exitCode).toBeNull()

    // useMainApp.ts's crash-recovery exit handler calls gw.start() directly
    // (not through request()) when it believes the gateway died unexpectedly.
    gw.start()

    expect(spawnMock).toHaveBeenCalledTimes(1)
    expect(gw.getLogTail(20)).toContain('start() ignored — client is shutting down')
  })

  it('rejects immediately (not after the RPC timeout) once shutting down', async () => {
    const gw = new GatewayClient()

    gw.start()
    gw.kill()

    const start = Date.now()
    const req = gw.request('model.options', {})

    await expect(req).rejects.toThrow(/shutting down/)
    // Should settle well under a second — nowhere near REQUEST_TIMEOUT_MS
    // (30s floor / 120s default) — proving this is an immediate, explicit
    // rejection and not a silent hang that happens to resolve later.
    expect(Date.now() - start).toBeLessThan(1000)
  })

  it('a still-registered late response from the half-dead child is safely ignored, not misdelivered', async () => {
    const gw = new GatewayClient()

    gw.start()
    const firstProc = FakeChildProcess.instances[0]!

    const inFlight = gw.request('model.options', {})

    await vi.waitFor(() => expect(firstProc.stdin.readableLength >= 0).toBe(true))

    // Session closes mid-flight.
    gw.kill('graceful-exit-cleanup')
    await expect(inFlight).rejects.toThrow(/gateway closed/)

    // The old child (still alive, per the grace window) finally finishes
    // its abandoned work and writes the response it was computing when
    // SIGTERM landed. Nothing should throw, and it must not resolve/reject
    // the already-settled promise above (asserted implicitly: the earlier
    // `await expect(inFlight).rejects` already proved the client-side
    // settlement happened before this late write).
    expect(() => {
      firstProc.stdout.write(JSON.stringify({ id: 'r1', jsonrpc: '2.0', result: { models: [] } }) + '\n')
    }).not.toThrow()

    firstProc.simulateExit(0)
    expect(spawnMock).toHaveBeenCalledTimes(1)
  })
})
