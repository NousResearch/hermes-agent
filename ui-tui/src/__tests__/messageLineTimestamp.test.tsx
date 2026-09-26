import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { fmtMsgTimestamp, MessageLine } from '../components/messageLine.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const userMsg = (overrides: Partial<Msg> = {}): Msg =>
  ({
    createdAt: 1_756_000_000,
    role: 'user',
    text: 'hello',
    ...overrides
  }) as Msg

function renderLines(msg: Msg, timestamps = true, cols = 60): string[] {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: cols, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(<MessageLine cols={cols} msg={msg} t={DEFAULT_THEME} timestamps={timestamps} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output).split('\n')
}

describe('MessageLine timestamp layout', () => {
  it('renders the timestamp beside the message instead of on its own row', () => {
    const msg = userMsg()
    const stamp = fmtMsgTimestamp(msg.createdAt)
    expect(stamp).not.toBeNull()

    const lines = renderLines(msg)
    expect(lines.some(line => line.includes(stamp!) && line.includes('hello'))).toBe(true)
    expect(lines.some(line => line.trim() === stamp)).toBe(false)
  })

  it('renders no timestamp when the message has no timestamp', () => {
    const lines = renderLines(userMsg({ createdAt: undefined }))

    expect(lines.some(line => /\[\d{2}:\d{2}\]/.test(line))).toBe(false)
    expect(lines.some(line => line.includes('hello'))).toBe(true)
  })

  it('stays bounded on a narrow terminal while keeping the timestamp inline', () => {
    const msg = userMsg({ text: 'abcdefghij' })
    const stamp = fmtMsgTimestamp(msg.createdAt)
    expect(stamp).not.toBeNull()

    const cols = 14
    const lines = renderLines(msg, true, cols).filter(Boolean)
    const stampLine = lines.find(line => line.includes(stamp!))

    expect(stampLine).toBeDefined()
    expect(stampLine!.trim()).not.toBe(stamp)
    expect(lines.every(line => line.length <= cols)).toBe(true)
  })
})
