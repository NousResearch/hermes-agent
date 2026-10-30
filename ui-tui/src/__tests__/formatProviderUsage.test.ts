import { describe, expect, it } from 'vitest'

import type { AccountUsageMeterAccount, AccountUsageMeterInfo } from '../app/interfaces.js'
import { formatProviderUsage } from '../components/appChrome.js'

const win = (session: number, week: number) => [
  { label: 'Current session', usedPercent: session },
  { label: 'Current week', usedPercent: week },
]

const account = (overrides: Partial<AccountUsageMeterAccount> & { label: string }): AccountUsageMeterAccount => ({
  active: false,
  available: true,
  windows: win(10, 20),
  ...overrides,
})

const provider = (overrides: Partial<AccountUsageMeterInfo> = {}): AccountUsageMeterInfo => ({
  available: true,
  windows: win(10, 20),
  ...overrides,
})

describe('formatProviderUsage', () => {
  it('renders bare percentages for a single-account pool', () => {
    expect(formatProviderUsage('Claude', provider())).toBe('Claude 10%/20%')
    expect(formatProviderUsage('Claude', provider({ accounts: [account({ label: 'only' })] }))).toBe('Claude 10%/20%')
  })

  it('renders a dash when the single provider is unavailable', () => {
    expect(formatProviderUsage('Codex', provider({ available: false, windows: [] }))).toBe('Codex —')
  })

  it('lists live accounts with the active marker and collapses dead ones to a count', () => {
    const text = formatProviderUsage(
      'Claude',
      provider({
        accounts: [
          account({ label: 'cg.chae-claude', windows: win(5, 14) }),
          account({ label: 'ANTHROPIC_TOKEN', available: false, windows: [], unavailableReason: 'HTTP 403' }),
          account({ label: 'CLAUDE_CODE_OAUTH_TOKEN', available: false, windows: [], unavailableReason: 'no token' }),
          account({ label: 'claude_code', active: true, windows: win(23, 32) }),
        ],
      }),
    )
    expect(text).toBe('Claude cg.chae-claude 5%/14% ●claude_code 23%/32% ⚠2')
    expect(text).not.toContain('ANTHROPIC_TOKEN')
    expect(text).not.toContain('HTTP 403')
  })

  it('treats an available account with no windows as unreadable (web contract parity)', () => {
    const text = formatProviderUsage(
      'Claude',
      provider({
        accounts: [
          account({ label: 'ghost', available: true, windows: [] }),
          account({ label: 'real', active: true, windows: win(1, 2) }),
        ],
      }),
    )
    expect(text).toBe('Claude ●real 1%/2% ⚠1')
    expect(text).not.toMatch(/ghost\s/)
  })

  it('falls back to a dash with the dead count when every account is unreadable', () => {
    const text = formatProviderUsage(
      'Claude',
      provider({
        accounts: [
          account({ label: 'a', available: false, windows: [] }),
          account({ label: 'b', available: true, windows: [] }),
        ],
      }),
    )
    expect(text).toBe('Claude ⚠2')
  })

  it('omits the warning suffix when every account is live', () => {
    const text = formatProviderUsage(
      'Claude',
      provider({ accounts: [account({ label: 'a', windows: win(1, 2) }), account({ label: 'b', active: true, windows: win(3, 4) })] }),
    )
    expect(text).toBe('Claude a 1%/2% ●b 3%/4%')
    expect(text).not.toContain('⚠')
  })
})
