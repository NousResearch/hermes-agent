import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { describe, expect, it, vi } from 'vitest'

import { ApprovalPrompt } from '../components/prompts.js'
import { DEFAULT_THEME } from '../theme.js'

class FakeInput extends EventEmitter {
  chunks: string[] = []
  isRaw = false
  isTTY = true
  readableLength = 0
  ref = vi.fn()
  setEncoding = vi.fn()
  unref = vi.fn()

  read() {
    const next = this.chunks.shift() ?? null
    this.readableLength = this.chunks.length

    return next
  }

  send(chunk: string) {
    this.chunks.push(chunk)
    this.readableLength = this.chunks.length
    this.emit('readable')
  }

  setRawMode = vi.fn((enabled: boolean) => {
    this.isRaw = enabled
  })
}

const settle = (ms = 25) => new Promise(resolve => setTimeout(resolve, ms))

describe('ApprovalPrompt long commands', () => {
  it('pages to every line of the command before a choice, and says where the rest is', async () => {
    const command = [...Array.from({ length: 10 }, (_, i) => `echo step ${i + 1}`), 'rm -rf ~/important'].join('\n')
    const onChoice = vi.fn()
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 40 })
    const frames: string[] = []
    stdout.on('data', chunk => frames.push(chunk.toString()))

    const instance = renderSync(
      <ApprovalPrompt
        cols={80}
        onChoice={onChoice}
        req={{ command, description: 'recursive delete', requestId: 'r1' }}
        t={DEFAULT_THEME}
      />,
      {
        patchConsole: false,
        stdin: stdin as unknown as NodeJS.ReadStream,
        stdout: stdout as unknown as NodeJS.WriteStream
      }
    )

    await settle()
    const first = stripAnsi(frames.join(''))
    expect(first).not.toContain('rm -rf ~/important')
    expect(first).toContain('lines 1–10 of 11 · press v to see the rest before choosing')
    expect(first).not.toContain('full text above')

    frames.length = 0
    stdin.send('v')
    await settle()
    const second = stripAnsi(frames.join(''))
    expect(second).toContain('rm -rf ~/important')
    expect(second).toContain('lines 11–11 of 11 (end)')
    expect(onChoice).not.toHaveBeenCalled()

    instance.unmount()
    instance.cleanup()
  })
})
