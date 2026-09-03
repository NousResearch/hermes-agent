import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { ClarifyPrompt } from '../components/prompts.js'
import { DEFAULT_THEME } from '../theme.js'

vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal<typeof import('@hermes/ink')>()

  return { ...mod, useInput: () => {} }
})

describe('ClarifyPrompt picker rows', () => {
  it('lists choices without numbered prefixes', () => {
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
      <ClarifyPrompt
        cols={80}
        onAnswer={() => {}}
        onCancel={() => {}}
        req={{ choices: ['Scope A', 'Scope B'], question: 'How scope?', requestId: 'r1' }}
        t={DEFAULT_THEME}
      />,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    const out = output.replace(/\u001b\[[0-9;]*m/g, '')
    expect(out).toMatch(/Scope A/)
    expect(out).toMatch(/Scope B/)
    expect(out).not.toMatch(/1\. Scope A/)
    expect(out).not.toMatch(/2\. Scope B/)

    instance.unmount()
    instance.cleanup()
  })
})
