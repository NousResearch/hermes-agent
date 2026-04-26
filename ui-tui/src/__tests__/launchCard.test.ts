import { describe, expect, it } from 'vitest'

import { logo } from '../banner.js'
import { buildLaunchCardModel, MOTTO } from '../components/branding.js'
import { DEFAULT_THEME } from '../theme.js'
import type { SessionInfo } from '../types.js'

const info: SessionInfo = {
  cwd: '/home/inu/agents-of-proto',
  mcp_servers: [
    { connected: true, name: 'chorus', tools: 85, transport: 'stdio' },
    { connected: true, name: 'context7', tools: 6, transport: 'stdio' },
    { connected: false, name: 'broken', tools: 0, transport: 'stdio' }
  ],
  model: 'openai/gpt-5.5',
  release_date: '2026.4.23',
  skills: {
    creative: ['ascii-art', 'interface-design-hermes'],
    'software-development': ['test-driven-development', 'session-context-verification'],
    mcp: ['chorus-coordination']
  },
  tools: {
    browser: ['browser_navigate', 'browser_snapshot'],
    file: ['read_file', 'write_file', 'patch'],
    terminal: ['terminal', 'process']
  },
  version: '0.11.0'
}

describe('Aurora launch card', () => {
  it('uses the new bridge/engine motto in the graffiti logo', () => {
    expect(MOTTO).toBe('The bridge remembers. The engine dreams.')
    expect(logo(DEFAULT_THEME.color).map(([, text]) => text).join('\n')).toContain('AURORA')
  })

  it('collapses tools and skills into cockpit counts instead of dumping inventories', () => {
    const model = buildLaunchCardModel(info, 'f7ef023f')
    const text = model.lines.map(line => line.text).join('\n')

    expect(text).toContain('☤ Aurora Proto')
    expect(text).toContain('The bridge remembers. The engine dreams.')
    expect(text).toContain('⚙ tools')
    expect(text).toContain('7 indexed across 3 toolsets')
    expect(text).toContain('✦ skills')
    expect(text).toContain('5 procedures across 3 rings')
    expect(text).toContain('/palette command bridge')
    expect(text).not.toContain('Available Tools')
    expect(text).not.toContain('browser_navigate')
    expect(text).not.toContain('test-driven-development')
  })

  it('renders a thread re-entry card for resumed sessions', () => {
    const model = buildLaunchCardModel(
      {
        ...info,
        resume_message_count: 42,
        resumed_session_id: '20260426_064219_1d59e7'
      },
      'abcd1234'
    )

    const text = model.lines.map(line => line.text).join('\n')

    expect(model.mode).toBe('resume')
    expect(text).toContain('THREAD RE-ENTRY')
    expect(text).toContain('◇ trace')
    expect(text).toContain('20260426_064219_1d59e7')
    expect(text).toContain('42 messages restored')
    expect(text).toContain('/resume thread')
  })
})
