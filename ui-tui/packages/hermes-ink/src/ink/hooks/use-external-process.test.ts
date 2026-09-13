import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import instances from '../instances.js'

// Mock React's useCallback so the test can import the module without React rendering.
vi.mock('react', () => ({
  useCallback: <T extends (...args: never[]) => unknown>(fn: T) => fn
}))

// Capture the bytes written to stdout so the assertion can inspect the
// terminal-control sequences emitted by withInkSuspended.
// withInkSuspended calls `process.stdout.write` directly (not via the
// Ink instance), so we patch the real process.stdout.write for the
// duration of each test.
let originalWrite: typeof process.stdout.write
let stdoutChunks: Buffer[]

function installStdoutSpy() {
  stdoutChunks = []
  originalWrite = process.stdout.write.bind(process.stdout)
  process.stdout.write = ((chunk: string | Uint8Array, ...rest: unknown[]) => {
    stdoutChunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
    return true
  }) as typeof process.stdout.write
}

function restoreStdout() {
  if (originalWrite) {
    process.stdout.write = originalWrite
  }
}

let enterAlternateScreen: ReturnType<typeof vi.fn>
let exitAlternateScreen: ReturnType<typeof vi.fn>
let fakeStdout: NodeJS.WriteStream

beforeEach(() => {
  installStdoutSpy()
  enterAlternateScreen = vi.fn()
  exitAlternateScreen = vi.fn()
  // Use process.stdout as the key so `instances.get(process.stdout)`
  // inside withInkSuspended returns our fake Ink instance.
  fakeStdout = process.stdout
  instances.set(fakeStdout, {
    enterAlternateScreen,
    exitAlternateScreen
  } as never)
})

afterEach(() => {
  instances.delete(fakeStdout)
  restoreStdout()
})

// Importing the module under test AFTER the mock + instance are wired.
const { withInkSuspended } = await import('./use-external-process.js')

function writtenSoFar(): string {
  return Buffer.concat(stdoutChunks).toString('binary')
}

describe('withInkSuspended', () => {
  it('emits the show-cursor sequence (\\x1b[?25h) after the external process returns', async () => {
    await withInkSuspended(async () => {
      // External process body — no-op.
    })

    // The whole point of the bugfix: the legacy `exitAlternateScreen` ends
    // with `?25l` (hide) because Ink used to render a synthetic caret via
    // inverted cells. Since the "rely on native cursor for input" change
    // the input box parks the real cursor, so leaving it hidden strands
    // the user with no visible caret. withInkSuspended must re-show it
    // synchronously after exitAlternateScreen.
    expect(writtenSoFar()).toContain('\x1b[?25h')

    // exitAlternateScreen should still have run — the show-cursor write is
    // ADDITIVE, not a replacement for the existing cleanup.
    expect(exitAlternateScreen).toHaveBeenCalledTimes(1)
  })

  it('runs the external process between enterAlternateScreen and exitAlternateScreen', async () => {
    const order: string[] = []

    enterAlternateScreen.mockImplementation(() => order.push('enter'))
    exitAlternateScreen.mockImplementation(() => order.push('exit'))

    await withInkSuspended(async () => {
      order.push('run')
    })

    expect(order).toEqual(['enter', 'run', 'exit'])
  })

  it('still emits the show-cursor sequence when the external process throws', async () => {
    await expect(
      withInkSuspended(async () => {
        throw new Error('editor crashed')
      })
    ).rejects.toThrow('editor crashed')

    expect(writtenSoFar()).toContain('\x1b[?25h')
  })

  it('runs the external process without Ink teardown when no instance is registered', async () => {
    instances.delete(fakeStdout)
    stdoutChunks.length = 0

    let ran = false

    await withInkSuspended(async () => {
      ran = true
    })

    expect(ran).toBe(true)
    expect(enterAlternateScreen).not.toHaveBeenCalled()
    expect(exitAlternateScreen).not.toHaveBeenCalled()
    // Non-Ink path doesn't touch stdout either — bypass mode for shells / tests.
    expect(stdoutChunks).toHaveLength(0)
  })
})
