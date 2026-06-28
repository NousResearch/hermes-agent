import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { SessionPanel } from '../components/branding.js'
import { DEFAULT_THEME } from '../theme.js'
import type { McpServerStatus, SessionInfo } from '../types.js'

// Invariant under test: the TUI banner's MCP headline counts *connected*
// servers, never configured-but-disabled ones. This mirrors the classic CLI
// banner (`mcp_connected = sum(1 for s in mcp_status if s["connected"])` in
// hermes_cli/banner.py) and the "connected" label on the MCP collapse toggle.
//
// Regression: branding.tsx used the raw `info.mcp_servers.length`, so a
// disabled `linear` server alongside a connected `nous-support` server made
// the TUI report "2 MCP" while the classic CLI correctly reported "1 MCP".

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const makeStreams = (columns = 100) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })

  let captured = ''
  stdout.on('data', chunk => {
    captured += chunk.toString()
  })

  return { capture: () => captured, stderr, stdin, stdout }
}

const mcp = (over: Partial<McpServerStatus> & Pick<McpServerStatus, 'name'>): McpServerStatus => ({
  connected: false,
  tools: 0,
  transport: 'http',
  ...over
})

const baseInfo = (mcp_servers: McpServerStatus[]): SessionInfo => ({
  mcp_servers,
  model: 'test-model',
  profiles: [
    {
      active: true,
      is_default: true,
      model: 'gpt-5.4',
      name: 'default',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'archived-codingcoco-20260609',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'audit-readonly',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'coding',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'critic',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'dev',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'executor',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'hermes-ops',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'planner',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'previewer',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'research',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'reviewer',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'sandbox',
      provider: 'openai-codex'
    },
    {
      model: 'gpt-5.4',
      name: 'scout',
      provider: 'openai-codex'
    },
    {
      gateway_running: true,
      model: 'gpt-5.4',
      name: 'telegram2',
      provider: 'openai-codex'
    },
    {
      gateway_running: true,
      model: 'gpt-5.4',
      name: 'telegram3',
      provider: 'openai-codex'
    },
    {
      gateway_running: true,
      model: 'gpt-5.4',
      name: 'jarvis_iii',
      provider: 'openai-codex'
    },
    {
      gateway_running: true,
      model: 'gpt-5.4',
      name: 'wifetodoist',
      provider: 'openai-codex'
    }
  ],
  skills: { core: ['a', 'b'] },
  tools: { file: ['read_file', 'write_file'] }
})

async function renderFooter(info: SessionInfo): Promise<string> {
  const streams = makeStreams()

  const instance = renderSync(React.createElement(SessionPanel, { info, sid: 'test', t: DEFAULT_THEME }), {
    patchConsole: false,
    stderr: streams.stderr as unknown as NodeJS.WriteStream,
    stdin: streams.stdin as unknown as NodeJS.ReadStream,
    stdout: streams.stdout as unknown as NodeJS.WriteStream
  })

  try {
    await delay(20)

    // Strip ANSI so we can assert on the rendered text content.
    // eslint-disable-next-line no-control-regex
    return streams.capture().replace(/\u001b\[[0-9;]*m/g, '')
  } finally {
    instance.unmount()
    instance.cleanup()
  }
}

describe('branding MCP headline count', () => {
  it('shows delegation profile names and hides archived or bot profiles', async () => {
    const frame = await renderFooter(baseInfo([]))

    expect(frame).toContain('Profiles (12)')
    for (const name of [
      'audit-readonly',
      'coding',
      'critic',
      'dev',
      'executor',
      'hermes-ops',
      'planner',
      'previewer',
      'research',
      'reviewer',
      'sandbox',
      'scout'
    ]) {
      expect(frame).toContain(name)
    }
    for (const name of ['default', 'archived-codingcoco', 'telegram2', 'telegram3', 'jarvis_iii', 'wifetodoist']) {
      expect(frame).not.toContain(name)
    }
    expect(frame).not.toContain('openai-codex')
    expect(frame).not.toContain('gpt-5.4')
    expect(frame).not.toContain('Available Tools')
  })

  it('counts only connected servers, not configured-but-disabled ones', async () => {
    const frame = await renderFooter(
      baseInfo([
        mcp({ connected: true, name: 'nous-support', status: 'connected', tools: 6 }),
        mcp({ connected: false, disabled: true, name: 'linear', status: 'disabled' })
      ])
    )

    // One connected server → "1 MCP", never "2 MCP".
    expect(frame).toContain('1 MCP')
    expect(frame).not.toContain('2 MCP')
  })

  it('drops the MCP segment entirely when no server is connected', async () => {
    const frame = await renderFooter(
      baseInfo([mcp({ connected: false, disabled: true, name: 'linear', status: 'disabled' })])
    )

    // Matches the classic CLI, which only appends "· N MCP" when N > 0.
    expect(frame).not.toContain('MCP servers')
    expect(frame).not.toMatch(/\d MCP\b/)
  })

  it('counts every connected server when several are connected', async () => {
    const frame = await renderFooter(
      baseInfo([
        mcp({ connected: true, name: 'alpha', status: 'connected' }),
        mcp({ connected: true, name: 'beta', status: 'connected' }),
        mcp({ connected: false, disabled: true, name: 'gamma', status: 'disabled' })
      ])
    )

    expect(frame).toContain('2 MCP')
    expect(frame).not.toContain('3 MCP')
  })
})
