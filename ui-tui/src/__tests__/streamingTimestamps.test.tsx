// Streaming messages must render their [HH:MM] timestamp while live, not only
// once settled — otherwise the timestamp pops in when the message flushes.
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { patchTurnState, resetTurnState } from '../app/turnStore.js'
import { StreamingAssistant } from '../components/streamingAssistant.js'

const flushEffects = async () => {
  for (let i = 0; i < 10; i++) {
    await new Promise(resolve => setTimeout(resolve, 5))
  }
}

const mountStreaming = (timestamps?: boolean) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <StreamingAssistant
      cols={80}
      detailsMode="collapsed"
      detailsModeCommandOverride={false}
      progress={{ showProgressArea: false }}
      {...(timestamps === undefined ? {} : { timestamps })}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  return { instance, text: () => stripAnsi(output) }
}

afterEach(() => {
  resetTurnState()
})

describe('StreamingAssistant — timestamps', () => {
  it('renders a [HH:MM] timestamp on the streaming assistant message when timestamps are on', async () => {
    patchTurnState({ streaming: 'Hello from the model' })

    const { instance, text } = mountStreaming(true)

    try {
      await flushEffects()
      expect(text()).toMatch(/\[\d{2}:\d{2}\]/)
    } finally {
      instance.unmount()
    }
  })

  it('renders no timestamp on the streaming assistant message when timestamps are off', async () => {
    patchTurnState({ streaming: 'Hello from the model' })

    const { instance, text } = mountStreaming(false)

    try {
      await flushEffects()
      expect(text()).not.toMatch(/\[\d{2}:\d{2}\]/)
    } finally {
      instance.unmount()
    }
  })
})
