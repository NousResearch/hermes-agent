import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { describe, expect, it, vi } from 'vitest'

import { VaultCodePrompt, VaultSaveLoginPrompt } from '../components/maskedPrompt.js'
import { DEFAULT_THEME } from '../theme.js'

class FakeInput extends EventEmitter {
  chunks: string[] = []
  isRaw = false
  isTTY = true
  readableLength = 0

  read() {
    const next = this.chunks.shift() ?? null
    this.readableLength = this.chunks.length

    return next
  }

  ref = vi.fn()

  send(...chunks: string[]) {
    this.chunks.push(...chunks)
    this.readableLength = this.chunks.length
    this.emit('readable')
  }

  setEncoding = vi.fn()

  setRawMode = vi.fn((enabled: boolean) => {
    this.isRaw = enabled
  })

  unref = vi.fn()
}

const settle = (ms = 25) => new Promise(resolve => setTimeout(resolve, ms))

function mount(onReady: (identifier: string, password: string) => void, node?: React.ReactElement) {
  const stdin = new FakeInput()
  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  const frames: string[] = []
  stdout.on('data', chunk => frames.push(chunk.toString()))

  const instance = renderSync(
    node ?? <VaultSaveLoginPrompt cols={80} onReady={onReady} site="www.linkedin.com" t={DEFAULT_THEME} />,
    {
      patchConsole: false,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    }
  )

  return { frames, instance, stdin, stdout }
}

const screenText = (frames: string[]) => stripAnsi(frames.join(''))

describe('VaultSaveLoginPrompt (#109101)', () => {
  it('shows the identifier step first for the requesting site', async () => {
    const onReady = vi.fn()
    const { frames, instance } = mount(onReady)

    await settle()
    expect(screenText(frames)).toContain('Save your www.linkedin.com login')
    expect(onReady).not.toHaveBeenCalled()

    instance.unmount()
    instance.cleanup()
  })

  it('declines the whole save when the identifier step is submitted empty', async () => {
    const onReady = vi.fn()
    const { instance, stdin } = mount(onReady)

    await settle()
    stdin.send('\r')
    await settle()

    expect(onReady).toHaveBeenCalledWith('', '')

    instance.unmount()
    instance.cleanup()
  })

  it('moves to the masked password step after a non-empty identifier, then emits both values', async () => {
    const onReady = vi.fn()
    const { frames, instance, stdin } = mount(onReady)

    await settle()
    stdin.send('me@x.io')
    await settle()
    stdin.send('\r')
    await settle()

    expect(screenText(frames)).toContain('Password for me@x.io')
    expect(onReady).not.toHaveBeenCalled()

    stdin.send('s3cret')
    await settle()
    stdin.send('\r')
    await settle()

    expect(onReady).toHaveBeenCalledWith('me@x.io', 's3cret')

    instance.unmount()
    instance.cleanup()
  })
})

describe('VaultCodePrompt', () => {
  it('shows the site and the backend hint, and submits the code as typed', async () => {
    const onSubmit = vi.fn()

    const { frames, instance, stdin } = mount(
      vi.fn(),
      <VaultCodePrompt cols={80} hint="sent to •••42" onSubmit={onSubmit} site="github.com" t={DEFAULT_THEME} />
    )

    await settle()
    stdin.send('123456')
    await settle()
    stdin.send('\r')
    await settle()

    const screen = screenText(frames)

    expect(screen).toContain('Verification code for github.com')
    expect(screen).toContain('sent to •••42')
    expect(screen).toContain('123456')
    expect(onSubmit).toHaveBeenCalledWith('123456')

    instance.unmount()
    instance.cleanup()
  })
})
