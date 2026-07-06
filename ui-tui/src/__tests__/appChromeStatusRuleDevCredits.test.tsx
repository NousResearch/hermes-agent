import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

// DEV_CREDITS_MODE is a module-load-time constant (config/env.ts reads
// process.env.HERMES_DEV_CREDITS exactly once, at import). Mutating process.env
// inside a test can't flip it after the module is loaded — so mock the module to
// the dev-on value for this file. vitest hoists vi.mock above the imports, so
// appChrome picks up the mocked flag. Lives in its own file so the override
// stays scoped (the other StatusRule tests run with the real, dev-off value).
vi.mock('../config/env.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../config/env.js')>()
  return { ...actual, DEV_CREDITS_MODE: true }
})

type ReactNodeLike = React.ReactNode

const textContent = (node: ReactNodeLike): string => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return ''
  }

  if (typeof node === 'string' || typeof node === 'number') {
    return String(node)
  }

  if (Array.isArray(node)) {
    return node.map(textContent).join('')
  }

  if (React.isValidElement(node)) {
    return textContent(node.props.children)
  }

  return ''
}

const baseProps = {
  bgCount: 0,
  busy: false,
  cols: 100,
  cwdLabel: '~/repo',
  liveSessionCount: 0,
  model: 'opus-4.8',
  sessionStartedAt: null,
  showCost: false,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, total: 50_000 },
  voiceLabel: ''
}

describe('StatusRule dev-credits banner (HERMES_DEV_CREDITS on)', () => {
  it('keeps the dev-credits banner visible alongside a notice', () => {
    const element = StatusRule({
      ...baseProps,
      notice: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' },
      usage: { ...baseProps.usage, dev_credits_spent_micros: 12_345 }
    })

    const rendered = textContent(element)

    // The notice and the dev banner coexist …
    expect(rendered).toContain('⚠ 90% used')
    expect(rendered).toContain('(dev credits)')
    // … and the Δ spend segment renders (12345 micros → 1.2¢).
    expect(rendered).toContain('Δ')
  })
})
