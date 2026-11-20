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

describe('ApprovalPrompt schedule card', () => {
  const render = (req: Parameters<typeof ApprovalPrompt>[0]['req']) => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 40 })
    const frames: string[] = []
    stdout.on('data', chunk => frames.push(chunk.toString()))

    const instance = renderSync(
      <ApprovalPrompt cols={100} onChoice={vi.fn()} onTell={vi.fn()} req={req} t={DEFAULT_THEME} />,
      {
        patchConsole: false,
        stdin: stdin as unknown as NodeJS.ReadStream,
        stdout: stdout as unknown as NodeJS.WriteStream
      }
    )

    return { frames, instance }
  }

  it('shows the job description and schedule labels, never the synthetic command; commands keep their card', async () => {
    const req = {
      command: '<cronjob_manage> (schedule a new job)',
      description: 'Schedule "water" (every 2h, first run 10:00 AM PST): remind me to drink water',
      requestId: 'r1'
    }

    const schedule = render({ ...req, kind: 'schedule' })
    await settle()
    const card = stripAnsi(schedule.frames.join(''))
    expect(card).toContain('⏰ schedule this job?')
    expect(card).toContain('remind me to drink water')
    expect(card).not.toContain('<cronjob_manage>')
    expect(card).not.toContain('approval required')

    for (const label of ['1. Allow', '2. Allow for this chat', '3. Always allow new jobs', '4. Deny']) {
      expect(card).toContain(label)
    }

    schedule.instance.unmount()
    schedule.instance.cleanup()

    for (const kind of [undefined, 'something-new']) {
      const command = render({ ...req, kind })
      await settle()
      const plain = stripAnsi(command.frames.join(''))
      expect(plain).toContain('⚠ approval required')
      expect(plain).toContain('<cronjob_manage> (schedule a new job)')
      expect(plain).toContain('1. Allow once')
      command.instance.unmount()
      command.instance.cleanup()
    }
  })
})
