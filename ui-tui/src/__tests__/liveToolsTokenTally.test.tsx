import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { patchTurnState } from '../app/turnStore.js'
import { StreamingAssistant } from '../components/streamingAssistant.js'
import type { ActiveTool } from '../types.js'

const flushEffects = async () => {
  // Passive effects + the re-render they trigger need a few macrotask turns
  // (React's scheduler uses MessageChannel) before the next frame paints.
  for (let i = 0; i < 10; i++) {
    await new Promise(resolve => setTimeout(resolve, 5))
  }
}

const mountLive = async () => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 30 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <StreamingAssistant
      cols={100}
      compact={false}
      detailsMode="expanded"
      detailsModeCommandOverride={false}
      progress={{ showProgressArea: true }}
      sections={{ thinking: 'hidden', tools: 'expanded' }}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  await flushEffects()

  return { instance, output: () => stripAnsi(output) }
}

const activeTool: ActiveTool = { context: 'src/app/turnController.ts', id: 'tool-1', name: 'read_file' }

afterEach(() => {
  patchTurnState({ streaming: '', toolTokens: 0, tools: [] })
})

describe('live tools panel — per-call token tally', () => {
  it('shows the running tally on the tools panel header while tools run', async () => {
    patchTurnState({ toolTokens: 42, tools: [activeTool] })

    const { instance, output } = await mountLive()

    expect(output()).toContain('~42 tokens')
    instance.unmount()
  })

  it('renders no tally before the first tool is charged', async () => {
    patchTurnState({ tools: [activeTool] })

    const { instance, output } = await mountLive()

    expect(output()).not.toContain('tokens')
    instance.unmount()
  })
})
