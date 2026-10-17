import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { SessionPanel } from '../components/branding.js'
import { messages } from '../i18n/runtime.js'
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
  skills: { core: ['a', 'b'] },
  tools: { file: ['read_file', 'write_file'] }
})

async function renderFooter(info: SessionInfo, t = DEFAULT_THEME): Promise<string> {
  const streams = makeStreams()
  // SessionPanel reads its width from process.stdout (useStdout), not the render
  // stream. Pin it to the stream's 100 columns so the layout doesn't depend on
  // the terminal the tests run in.
  const columns = Object.getOwnPropertyDescriptor(process.stdout, 'columns')
  Object.defineProperty(process.stdout, 'columns', { configurable: true, value: 100 })

  const instance = renderSync(React.createElement(SessionPanel, { info, sid: 'test', t }), {
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

    if (columns) {
      Object.defineProperty(process.stdout, 'columns', columns)
    } else {
      delete (process.stdout as { columns?: number }).columns
    }
  }
}

describe('branding MCP headline count', () => {
  it('counts only connected servers, not configured-but-disabled ones', async () => {
    const frame = await renderFooter(
      baseInfo([
        mcp({ connected: true, name: 'nous-support', status: 'connected', tools: 6 }),
        mcp({ connected: false, disabled: true, name: 'linear', status: 'disabled' })
      ])
    )

    // One connected server → "1 MCP", never "2 MCP".
    expect(frame).toContain(messages().chatBits.branding.mcpSummary(1))
    expect(frame).not.toContain(messages().chatBits.branding.mcpSummary(2))
  })

  it('uses one full-width metadata column when a skin suppresses the panel hero', async () => {
    const frame = await renderFooter(baseInfo([]), { ...DEFAULT_THEME, bannerHero: ' ' })

    expect(frame).toContain('test-model · Nous Research')
    expect(frame).toContain(`${messages().chatBits.branding.sessionLabel}test`)
  })

  it('keeps a narrow hero from squeezing the model and session lines', async () => {
    const frame = await renderFooter(baseInfo([]), { ...DEFAULT_THEME, bannerHero: '[#FFD700]★[/]' })

    expect(frame).toContain('★')
    expect(frame).toContain('test-model · Nous Research')
    expect(frame).toContain(`${messages().chatBits.branding.sessionLabel}test`)
  })

  it('drops the MCP segment entirely when no server is connected', async () => {
    const frame = await renderFooter(
      baseInfo([mcp({ connected: false, disabled: true, name: 'linear', status: 'disabled' })])
    )

    // Matches the classic CLI, which only appends "· N MCP" when N > 0.
    expect(frame).not.toContain('MCP servers')
    expect(frame).not.toMatch(/\d MCP\b/)
  })
})
