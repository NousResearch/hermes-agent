import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { UserInputPrompt } from '../components/prompts.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

function renderToText(node: React.ReactElement) {
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

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

describe('UserInputPrompt', () => {
  it('renders native context, question count, options, and pending guidance', () => {
    const output = renderToText(
      <UserInputPrompt
        cols={80}
        onAnswer={async () => true}
        onCancel={() => undefined}
        req={{
          context: 'Pick the implementation path',
          expiresAt: 0,
          questions: [{ allowFreeText: true, id: 'path', options: ['A', 'B'], text: 'Which path?' }],
          requestId: 'input-1',
          sessionId: 's1'
        }}
        t={DEFAULT_THEME}
      />
    )

    expect(output).toContain('input 1 question')
    expect(output).toContain('Pick the implementation path')
    expect(output).toContain('Which path?')
    expect(output).toContain('1. A')
    expect(output).toContain('2. B')
    expect(output).toContain('Esc leave pending')
  })
})
