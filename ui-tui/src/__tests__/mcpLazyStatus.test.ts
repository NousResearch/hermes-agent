import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { McpServerLine, SessionPanel } from '../components/branding.js'
import { DEFAULT_THEME } from '../theme.js'
import type { McpServerStatus, SessionInfo } from '../types.js'

// Invariant under test: `lazy` is a Python→TUI wire value.
// `tools/mcp_tool_discovery.get_mcp_status()` emits `status: "lazy"` for a server
// registered from the schema cache whose process has not been spawned, and
// `tui_gateway/server.py` forwards that list verbatim as `info["mcp_servers"]`.
//
// Two things must hold on this side of the wire:
//   1. `McpServerStatus.status` can represent it — otherwise the runtime protocol
//      and its typed consumer have drifted. The compile-time guard for that is the
//      `s.status === 'lazy'` comparison in branding.tsx: narrowing the union back to
//      the old five values makes `tsc` fail it with TS2367. (`src/__tests__` is
//      excluded from tsconfig.json, so this file is never part of that program —
//      what it pins is the runtime payload shape.)
//   2. The row renders the cached tool count. The status chain ends in an
//      unconditional red "failed", so before the `lazy` branch existed a healthy
//      lazy server rendered as *failed* — the same misreport the CLI banner fix
//      addresses, reproduced on the TUI surface.

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

async function renderNode(node: React.ReactElement): Promise<string> {
  const streams = makeStreams()

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: streams.stderr as NodeJS.WriteStream,
    stdin: streams.stdin as NodeJS.ReadStream,
    stdout: streams.stdout as NodeJS.WriteStream
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

// Exactly the payload get_mcp_status() produces for a lazy server: never connected,
// not disabled, carrying the cached tool count.
const LAZY: McpServerStatus = {
  connected: false,
  disabled: false,
  name: 'playwright',
  status: 'lazy',
  tools: 3,
  transport: 'stdio'
}

describe('lazy MCP server status', () => {
  it('is representable by SessionInfo and keeps connected false with its cached count', () => {
    // A lazy server is carried by SessionInfo unchanged: never connected, not disabled,
    // and keeping the tool count discovery cached for it.
    const info: SessionInfo = {
      mcp_servers: [LAZY],
      model: 'test-model',
      skills: {},
      tools: {}
    }

    expect(info.mcp_servers?.[0].status).toBe('lazy')
    expect(info.mcp_servers?.[0].connected).toBe(false)
    expect(info.mcp_servers?.[0].tools).toBe(3)
  })

  it('renders the cached tool count, not the red failed default', async () => {
    const frame = await renderNode(React.createElement(McpServerLine, { s: LAZY, t: DEFAULT_THEME }))

    expect(frame).toContain('playwright')
    expect(frame).toContain('3 tools')
    expect(frame).toContain('(lazy)')
    expect(frame).not.toContain('failed')
    expect(frame).not.toContain('configured')
  })

  it('is not counted in the connected MCP headline', async () => {
    // Parity with hermes_cli/banner.py, whose headline is sum(s["connected"]):
    // a lazy server has no live session, so it must not inflate "N MCP".
    const info: SessionInfo = {
      mcp_servers: [LAZY, { connected: true, name: 'nous-support', status: 'connected', tools: 6, transport: 'http' }],
      model: 'test-model',
      skills: {},
      tools: {}
    }

    const frame = await renderNode(React.createElement(SessionPanel, { info, sid: 'test', t: DEFAULT_THEME }))

    expect(frame).toContain('1 MCP')
    expect(frame).not.toContain('2 MCP')
  })
})
