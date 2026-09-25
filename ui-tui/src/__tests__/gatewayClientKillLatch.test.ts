import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The race from issue #114987 lives only in spawn mode: kill() leaves
// `this.proc` pointing at the child it just killed, so the late `exit` event
// passes the identity guard, reaches handleTransportExit and is emitted to
// the app — whose recovery subscriber answers with start(), which un-latches
// `disposed` and spawns a replacement gateway onto the vanished PTY. These
// tests drive that exact sequence with a fake child_process.spawn.

const { fakeSpawn, FakeChildProcess } = vi.hoisted(() => {
  class FakeStream {
    private listeners = new Map<string, Array<(event: any) => void>>()

    resume() {
      return this
    }

    pause() {
      return this
    }

    write(_text: string) {
      return true
    }

    on(type: string, callback: (event: any) => void) {
      const entries = this.listeners.get(type) ?? []

      entries.push(callback)
      this.listeners.set(type, entries)
    }

    removeListener(type: string, callback: (event: any) => void) {
      const entries = this.listeners.get(type)

      if (!entries) {
        return
      }

      this.listeners.set(
        type,
        entries.filter(entry => entry !== callback)
      )
    }

    emit(type: string, ...args: unknown[]) {
      for (const callback of [...(this.listeners.get(type) ?? [])]) {
        callback(...args)
      }
    }
  }

  class FakeChildProcess {
    static instances: FakeChildProcess[] = []

    killed = false
    exitCode: null | number = null
    signalCode: null | string = null
    pid = 4242
    stdin = new FakeStream()
    stdout = new FakeStream()
    stderr = new FakeStream()
    private listeners = new Map<string, Array<(event: any) => void>>()

    constructor() {
      FakeChildProcess.instances.push(this)
    }

    kill() {
      this.killed = true

      return true
    }

    on(type: string, callback: (event: any) => void) {
      const entries = this.listeners.get(type) ?? []

      entries.push(callback)
      this.listeners.set(type, entries)
    }

    emit(type: string, ...args: unknown[]) {
      for (const callback of [...(this.listeners.get(type) ?? [])]) {
        callback(...args)
      }
    }
  }

  const fakeSpawn = vi.fn(() => new FakeChildProcess())

  return { fakeSpawn, FakeChildProcess }
})

vi.mock('node:child_process', () => ({ spawn: fakeSpawn }))
vi.mock('node:fs', () => ({ existsSync: vi.fn(() => false) }))

import { GatewayClient } from '../gatewayClient.js'

describe('GatewayClient spawn-mode kill latch (issue #114987)', () => {
  let originalGatewayUrl: string | undefined
  let originalSidecarUrl: string | undefined

  beforeEach(() => {
    originalGatewayUrl = process.env.HERMES_TUI_GATEWAY_URL
    originalSidecarUrl = process.env.HERMES_TUI_SIDECAR_URL
    delete process.env.HERMES_TUI_GATEWAY_URL
    delete process.env.HERMES_TUI_SIDECAR_URL
    fakeSpawn.mockClear()
    FakeChildProcess.instances.length = 0
  })

  afterEach(() => {
    if (originalGatewayUrl === undefined) {
      delete process.env.HERMES_TUI_GATEWAY_URL
    } else {
      process.env.HERMES_TUI_GATEWAY_URL = originalGatewayUrl
    }

    if (originalSidecarUrl === undefined) {
      delete process.env.HERMES_TUI_SIDECAR_URL
    } else {
      process.env.HERMES_TUI_SIDECAR_URL = originalSidecarUrl
    }

    fakeSpawn.mockClear()
    FakeChildProcess.instances.length = 0
  })

  it('a killed child late exit does not respawn a replacement gateway', async () => {
    const gw = new GatewayClient()
    const exits: Array<null | number> = []

    // The recovery subscriber useMainApp installs: an emitted 'exit' with a
    // session to recover restarts the gateway.
    gw.on('exit', code => {
      exits.push(code)

      if (exits.length === 1) {
        gw.start()
      }
    })

    gw.start()
    gw.drain()
    await Promise.resolve()
    expect(fakeSpawn).toHaveBeenCalledTimes(1)

    // Intentional kill (graceful-exit cleanup, dead PTY): the reference must
    // be detached before the kill so the late exit is identity-skipped and
    // the recovery subscriber never sees an 'exit' to restart from.
    gw.kill('graceful-exit-cleanup')
    FakeChildProcess.instances[0]!.emit('exit', null, 'SIGTERM')

    expect(exits).toEqual([])
    expect(fakeSpawn).toHaveBeenCalledTimes(1)
  })

  it('start() after kill() is refused even when called directly', () => {
    const gw = new GatewayClient()

    gw.start()
    expect(fakeSpawn).toHaveBeenCalledTimes(1)

    gw.kill('app.die')
    gw.start()
    expect(fakeSpawn).toHaveBeenCalledTimes(1)
  })

  it('an unexpected child death still respawns through the recovery subscriber', async () => {
    const gw = new GatewayClient()
    const exits: Array<null | number> = []

    gw.on('exit', code => {
      exits.push(code)

      if (exits.length === 1) {
        gw.start()
      }
    })

    gw.start()
    gw.drain()
    await Promise.resolve()

    // Crash while the TUI is alive: identity intact, exit must be emitted.
    FakeChildProcess.instances[0]!.emit('exit', 1, null)

    expect(exits).toEqual([1])
    expect(fakeSpawn).toHaveBeenCalledTimes(2)
  })
})
