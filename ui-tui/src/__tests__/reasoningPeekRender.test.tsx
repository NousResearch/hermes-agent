import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import { DEFAULT_THEME } from '../theme.js'

const renderMessage = async (reasoningPeek: boolean) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <MessageLine
      cols={80}
      detailsMode="collapsed"
      msg={{
        kind: 'trail',
        role: 'system',
        text: '',
        thinking: 'peeked-live-reasoning',
        thinkingTokens: 4,
        isLiveReasoning: true
      }}
      reasoningActive
      reasoningPeek={reasoningPeek}
      sections={{ thinking: 'hidden' }}
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  // Flush mount effects before reading the frame (same settling as the MoA
  // visibility test -- reasoningAlwaysVisible seeds the open state on mount).
  await new Promise(resolve => setImmediate(resolve))
  await new Promise(resolve => setImmediate(resolve))

  const frame = stripAnsi(output)

  instance.unmount()
  instance.cleanup()

  return frame
}

describe('MessageLine -- per-turn reasoning peek render (#121979)', () => {
  it('renders the live reasoning segment under hidden thinking sections while the peek is active', async () => {
    const frame = await renderMessage(true)

    expect(frame).toContain('peeked-live')
    expect(frame).toContain('Thinking')
  })

  it('stays hidden without the peek even though the segment carries live reasoning', async () => {
    const frame = await renderMessage(false)

    expect(frame).not.toContain('peeked-live')
  })
})
