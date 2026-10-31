import { EventEmitter } from 'node:events'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TextInput } from '../components/textInput.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 80
  isRaw = false
  isTTY = true
  readableLength = 0
  rows = 24

  private buffered: string[] = []

  pushInput(chunk: string): void {
    this.buffered.push(chunk)
    this.readableLength += chunk.length
    this.emit('readable')
  }

  read(): null | string {
    const next = this.buffered.shift() ?? null
    this.readableLength = this.buffered.reduce((sum, chunk) => sum + chunk.length, 0)

    return next
  }

  ref(): this {
    return this
  }

  setEncoding(): this {
    return this
  }

  setRawMode(mode: boolean): this {
    this.isRaw = mode

    return this
  }

  unref(): this {
    return this
  }

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const tick = () => new Promise<void>(resolve => setImmediate(resolve))

function stubProcessStdout(writes: string[]): () => void {
  const originalWrite = process.stdout.write
  const originalIsTty = Object.getOwnPropertyDescriptor(process.stdout, 'isTTY')

  Object.defineProperty(process.stdout, 'isTTY', {
    configurable: true,
    value: true
  })

  process.stdout.write = ((
    chunk: string | Uint8Array,
    encodingOrCb?: BufferEncoding | ((err?: Error | null) => void),
    cb?: (err?: Error | null) => void
  ) => {
    writes.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    const callback = typeof encodingOrCb === 'function' ? encodingOrCb : cb
    callback?.()

    return true
  }) as typeof process.stdout.write

  return () => {
    process.stdout.write = originalWrite

    if (originalIsTty) {
      Object.defineProperty(process.stdout, 'isTTY', originalIsTty)
    } else {
      Reflect.deleteProperty(process.stdout, 'isTTY')
    }
  }
}

describe('TextInput unmount cleanup', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('flushes a deferred fast-echo parent change before unmount', async () => {
    vi.stubEnv('TERM_PROGRAM', 'vscode')
    vi.stubEnv('TERMUX_VERSION', '')
    vi.stubEnv('PREFIX', '')

    const stdout = new FakeTty()
    stdout.isTTY = false
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const changes: string[] = []
    const directWrites: string[] = []
    const restoreStdout = stubProcessStdout(directWrites)
    let didUnmount = false

    const instance = renderSync(
      React.createElement(TextInput, {
        columns: 80,
        onChange: value => changes.push(value),
        value: 'hel'
      }),
      {
        exitOnCtrlC: false,
        patchConsole: false,
        stderr: stderr as unknown as NodeJS.WriteStream,
        stdin: stdin as unknown as NodeJS.ReadStream,
        stdout: stdout as unknown as NodeJS.WriteStream
      }
    )

    try {
      await tick()
      await tick()

      expect(stdin.isRaw).toBe(true)
      expect(stdin.listenerCount('readable')).toBeGreaterThan(0)

      stdin.pushInput('p')

      expect(directWrites).toContain('p')
      expect(changes).toEqual([])

      instance.unmount()
      instance.cleanup()
      didUnmount = true

      expect(changes).toEqual(['help'])
    } finally {
      if (!didUnmount) {
        instance.unmount()
        instance.cleanup()
      }

      restoreStdout()
    }
  })
})
