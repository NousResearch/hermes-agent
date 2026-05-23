import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { DEFAULT_THEME } from '../theme.js'
import type { Theme } from '../theme.js'

const inputHandlers = vi.hoisted(() => [] as ((ch: string, key: { return?: boolean }) => void)[])

vi.mock('@hermes/ink', async importOriginal => {
  const actual = await importOriginal()

  return {
    ...actual,
    useInput: (handler: (ch: string, key: { return?: boolean }) => void) => inputHandlers.push(handler)
  }
})

const makeStreams = () => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', () => {})

  return { stderr, stdin, stdout }
}

describe('ApprovalPrompt', () => {
  it('denies by default when Enter is pressed without moving selection', async () => {
    inputHandlers.length = 0
    const onChoice = vi.fn()
    const streams = makeStreams()
    const { ApprovalPrompt } = await import('../components/prompts.js')

    const instance = renderSync(
      React.createElement(ApprovalPrompt, {
        onChoice,
        req: { command: 'rm -rf /tmp/example', description: 'dangerous command' },
        t: DEFAULT_THEME as Theme
      }),
      {
        patchConsole: false,
        stderr: streams.stderr as NodeJS.WriteStream,
        stdin: streams.stdin as NodeJS.ReadStream,
        stdout: streams.stdout as NodeJS.WriteStream
      }
    )

    try {
      expect(inputHandlers).toHaveLength(1)

      inputHandlers[0]!('', { return: true })

      expect(onChoice).toHaveBeenCalledWith('deny')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
