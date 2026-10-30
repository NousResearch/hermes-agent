import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { canonicalResult } from '../canonicalGateway.js'
import { SessionPanel } from '../components/branding.js'
import { DEFAULT_THEME } from '../theme.js'
import type { SessionInfo } from '../types.js'

// A canonical gateway's create/resume snapshot carries main's banner inventory
// (gateway/session_local_inventory.py). The TUI renders main's banner from it
// unchanged: tools by toolset, skills by category, MCP servers, and the
// "N tools · M skills · K MCP" footer — never a narrow-snapshot placeholder.

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const snapshot = {
  authority_epoch: 3,
  execution_generation: 0,
  info: {
    branch: 'main',
    cwd: '/work/repo',
    desktop_protocol: 'hermes-gateway-v1',
    lazy: true,
    mcp_servers: [
      { connected: true, name: 'nous-support', status: 'connected', tools: 6, transport: 'http' },
      { connected: false, disabled: true, name: 'linear', status: 'disabled', tools: 0, transport: 'stdio' }
    ],
    model: 'anthropic/claude-opus-5.5',
    profile_id: '/home/u/.hermes',
    profile_name: 'default',
    provider: 'nous',
    reasoning_effort: 'high',
    release_date: '2026.9.24',
    skills: { creative: ['ascii-art'], research: ['arxiv', 'blogwatcher'] },
    source: 'tui',
    tools: { file: ['patch', 'read_file', 'search_files', 'write_file'], terminal: ['process', 'terminal'] },
    version: '0.18.0'
  },
  last_sequence: 0,
  message_count: 0,
  messages: [],
  pending: [],
  prompts: [],
  replay_epoch: 'r',
  revision: 0,
  running: false,
  session_id: 'local-abc',
  stored_session_id: 'local-abc',
  subscription_id: 'sub'
}

async function render(info: SessionInfo): Promise<string> {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 140, isTTY: false, rows: 60 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  let captured = ''
  stdout.on('data', chunk => {
    captured += chunk.toString()
  })

  const instance = renderSync(React.createElement(SessionPanel, { info, sid: 'local-abc', t: DEFAULT_THEME }), {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  try {
    await delay(20)

    // eslint-disable-next-line no-control-regex
    return captured.replace(/\u001b\[[0-9;?]*[A-Za-z]/g, '')
  } finally {
    instance.unmount()
    instance.cleanup()
  }
}

describe('banner from a canonical session snapshot', () => {
  it('renders the tools, skills and MCP sections main showed', async () => {
    const { info } = canonicalResult('session.create', snapshot)
    const frame = await render(info)

    expect(frame).toContain('v0.18.0 (2026.9.24)')
    expect(frame).toContain('claude-opus-5.5')
    expect(frame).toContain('/work/repo')
    expect(frame).toMatch(/file: patch, read_file, search_files, write_file/)
    expect(frame).toMatch(/terminal: process, terminal/)
    expect(frame).toMatch(/Available Skills/)
    expect(frame).toMatch(/MCP Servers/)
    expect(frame).toMatch(/6 tools/)
    expect(frame).toMatch(/3 skills/)
    expect(frame).toMatch(/1 MCP/)
    expect(frame).not.toMatch(/not exposed by this runtime/)
  })

  it('renders main\'s banner, not a placeholder, before the inventory arrives', async () => {
    // A non-local route's snapshot info is `{ desktop_protocol }` only; main's banner still
    // draws its card (headings, version line slot, loading rows) rather than a one-line notice.
    const { info } = canonicalResult('session.resume', {
      ...snapshot,
      info: { desktop_protocol: 'hermes-gateway-v1', lazy: true, model: 'anthropic/claude-opus-5.5', version: '0.18.0' }
    })

    const frame = await render(info)

    expect(frame).toMatch(/Available Tools/)
    expect(frame).toMatch(/Available Skills/)
    expect(frame).toContain('v0.18.0')
    expect(frame).not.toMatch(/not exposed by this runtime/)
  })
})
