import { spawn } from 'node:child_process'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { shouldExitForSignal } from '../lib/gracefulExit.js'

type FatalScope = 'uncaughtException' | 'unhandledRejection'
type InstalledHandler = (...args: unknown[]) => void

interface InstallOptions {
  cleanups?: (() => Promise<void> | void)[]
  failsafeMs?: number
  ignoredSignals?: ('SIGHUP' | 'SIGINT' | 'SIGTERM')[]
  onError?: (scope: FatalScope, err: unknown) => void
  onSignal?: (signal: NodeJS.Signals) => void
}

interface ChildResult {
  code: number | null
  signal: NodeJS.Signals | null
  stderr: string
}

function runNodeScript(script: string): Promise<ChildResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['--import', 'tsx', '--input-type=module', '--eval', script], {
      stdio: ['ignore', 'ignore', 'pipe']
    })

    const stderr: Buffer[] = []

    child.stderr.on('data', chunk => stderr.push(Buffer.from(chunk)))

    const timeout = setTimeout(() => {
      child.kill('SIGKILL')
      reject(new Error('fatal-exit child did not stop within five seconds'))
    }, 5000)

    child.once('error', error => {
      clearTimeout(timeout)
      reject(error)
    })
    child.once('exit', (code, signal) => {
      clearTimeout(timeout)
      resolve({ code, signal, stderr: Buffer.concat(stderr).toString() })
    })
  })
}

async function installGracefulExit(options: InstallOptions = {}) {
  const handlers = new Map<string | symbol, InstalledHandler>()
  const onSpy = vi.spyOn(process, 'on')

  onSpy.mockImplementation(((event: string | symbol, listener: InstalledHandler) => {
    handlers.set(event, listener)

    return process
  }) as typeof process.on)

  const exitSpy = vi.spyOn(process, 'exit').mockImplementation((_code?: string | number | null) => undefined as never)

  const { setupGracefulExit } = await import('../lib/gracefulExit.js')

  setupGracefulExit(options)

  return {
    exitSpy,
    handler(event: string) {
      const installed = handlers.get(event)

      if (!installed) {
        throw new Error(`missing process handler for ${event}`)
      }

      return installed
    }
  }
}

beforeEach(() => {
  vi.resetModules()
  vi.useFakeTimers()
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('shouldExitForSignal', () => {
  it('ignores only the signals explicitly disabled for embedded dashboard chat', () => {
    expect(shouldExitForSignal('SIGINT', ['SIGINT'])).toBe(false)
    expect(shouldExitForSignal('SIGTERM', ['SIGINT'])).toBe(true)
    expect(shouldExitForSignal('SIGHUP', ['SIGINT'])).toBe(true)
  })
})

describe('setupGracefulExit', () => {
  it('keeps the failsafe alive when cleanup never finishes', async () => {
    vi.useRealTimers()

    const moduleUrl = new URL('../lib/gracefulExit.ts', import.meta.url).href

    const script = `
      import { setupGracefulExit } from ${JSON.stringify(moduleUrl)}
      setupGracefulExit({
        cleanups: [() => new Promise(() => {})],
        failsafeMs: 50
      })
      queueMicrotask(() => { throw new Error('fatal child error') })
    `

    const result = await runNodeScript(script)

    expect(result, result.stderr).toMatchObject({ code: 1, signal: null })
  })

  it.each([
    ['uncaughtException', new Error('fatal exception')],
    ['unhandledRejection', new Error('fatal rejection')]
  ] as const)('cleans up and exits 1 after %s even when the error hook throws', async (scope, error) => {
    const cleanup = vi.fn()

    const onError = vi.fn(() => {
      expect(vi.getTimerCount()).toBe(1)
      throw new Error('diagnostic hook failed')
    })

    const { exitSpy, handler } = await installGracefulExit({ cleanups: [cleanup], onError })

    handler(scope)(error)
    await vi.advanceTimersByTimeAsync(0)

    expect(onError).toHaveBeenCalledWith(scope, error)
    expect(cleanup).toHaveBeenCalledOnce()
    expect(exitSpy).toHaveBeenCalledWith(1)
  })

  it.each([
    ['SIGHUP', 129],
    ['SIGINT', 130],
    ['SIGTERM', 143]
  ] as const)('preserves the %s exit code when the signal hook throws', async (signal, exitCode) => {
    const cleanup = vi.fn()

    const onSignal = vi.fn(() => {
      expect(vi.getTimerCount()).toBe(1)
      throw new Error('signal hook failed')
    })

    const { exitSpy, handler } = await installGracefulExit({ cleanups: [cleanup], onSignal })

    handler(signal)()
    await vi.advanceTimersByTimeAsync(0)

    expect(onSignal).toHaveBeenCalledWith(signal)
    expect(cleanup).toHaveBeenCalledOnce()
    expect(exitSpy).toHaveBeenCalledWith(exitCode)
  })

  it('does not run hooks, cleanup, or exit for an ignored signal', async () => {
    const cleanup = vi.fn()
    const onSignal = vi.fn()

    const { exitSpy, handler } = await installGracefulExit({
      cleanups: [cleanup],
      ignoredSignals: ['SIGINT'],
      onSignal
    })

    handler('SIGINT')()
    await vi.advanceTimersByTimeAsync(0)

    expect(vi.getTimerCount()).toBe(0)
    expect(onSignal).not.toHaveBeenCalled()
    expect(cleanup).not.toHaveBeenCalled()
    expect(exitSpy).not.toHaveBeenCalled()
  })
})
