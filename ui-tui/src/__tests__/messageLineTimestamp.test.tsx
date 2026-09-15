import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { fmtMsgTimestamp, MessageLine } from '../components/messageLine.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

// The `display.timestamps` comment block on the stamp says the intent is a
// dim [HH:MM] *beside the gutter glyph* — one row per message, never a row
// of its own. These tests pin that layout contract: the stamp must share
// the gutter row with the message text.

const userMsg = (overrides: Partial<Msg> = {}): Msg =>
  ({
    createdAt: 1_756_000_000,
    role: 'user',
    text: 'hello',
    ...overrides
  }) as Msg

function renderLines(msg: Msg, timestamps?: boolean): string[] {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  let output = ''

  Object.assign(stdout, { columns: 60, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <MessageLine cols={60} msg={msg} t={DEFAULT_THEME} timestamps={timestamps} />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  instance.unmount()
  instance.cleanup()

  // A single static frame: split the stripped output into rows.
  return stripAnsi(output).split('\n')
}

describe('MessageLine timestamp layout', () => {
  it('renders the [HH:MM] stamp on the gutter row, not as its own row', () => {
    const msg = userMsg()
    const stamp = fmtMsgTimestamp(msg.createdAt)
    expect(stamp).not.toBeNull()

    const lines = renderLines(msg, true)
    const escapedStamp = stamp!.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

    // The stamp shares a row with the message text (beside the gutter glyph).
    expect(lines.some(line => line.includes(stamp!) && line.includes('hello'))).toBe(true)
    // ...and no row is just a gutter spacer plus the stamp.
    expect(lines.some(line => new RegExp(`^\\s*${escapedStamp}\\s*$`).test(line))).toBe(false)
  })

  it('renders no stamp at all when the message has no timestamp', () => {
    const lines = renderLines(userMsg({ createdAt: undefined }), true)

    expect(lines.some(line => /\[\d{2}:\d{2}\]/.test(line))).toBe(false)
    expect(lines.some(line => line.includes('hello'))).toBe(true)
  })
})
