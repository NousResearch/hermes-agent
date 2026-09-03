import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { SessionPanel } from '../components/branding.js'
import { DEFAULT_THEME } from '../theme.js'
import type { SessionInfo } from '../types.js'

const info: SessionInfo = {
  mcp_servers: [{ connected: true, name: 'nous-support', tools: 3, transport: 'http' }],
  model: 'anthropic/claude-sonnet-4.6',
  skills: { core: ['a'] },
  tools: { file: ['read_file'] }
}

const renderPanel = (compact: boolean) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    React.createElement(SessionPanel, { compact, info, sid: 'sess', t: DEFAULT_THEME }),
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  const text = () => output.replace(/\u001b\[[0-9;]*m/g, '')

  return { cleanup: () => instance.cleanup(), text, unmount: () => instance.unmount() }
}

describe('SessionPanel compact intro', () => {
  it('renders a one-line summary when compact', () => {
    const { text, unmount, cleanup } = renderPanel(true)
    const out = text()

    expect(out).toMatch(/claude-sonnet-4\.6/)
    expect(out).toMatch(/1 tools/)
    expect(out).toMatch(/1 skills/)
    expect(out).toMatch(/1 MCP/)
    expect(out).toMatch(/\/help/)
    expect(out).not.toMatch(/Available Tools/)

    unmount()
    cleanup()
  })

  it('keeps the full bordered panel when not compact', () => {
    const { text, unmount, cleanup } = renderPanel(false)
    const out = text()

    expect(out).toMatch(/Available Tools/)
    expect(out).toMatch(/Available Skills/)

    unmount()
    cleanup()
  })
})
