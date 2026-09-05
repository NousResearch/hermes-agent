import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

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

const findClickableWithText = (node: ReactNodeLike, needle: string): React.ReactElement | null => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return null
  }

  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findClickableWithText(child, needle)

      if (found) {
        return found
      }
    }

    return null
  }

  if (!React.isValidElement(node)) {
    return null
  }

  if (typeof node.props.onClick === 'function' && textContent(node).includes(needle)) {
    return node
  }

  return findClickableWithText(node.props.children, needle)
}

// Find the innermost element whose own (direct) text content includes the
// needle. Used to assert the colour the notice text is rendered with.
const findElementWithText = (node: ReactNodeLike, needle: string): React.ReactElement | null => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return null
  }

  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findElementWithText(child, needle)

      if (found) {
        return found
      }
    }

    return null
  }

  if (!React.isValidElement(node)) {
    return null
  }

  // Prefer the deepest matching element so we get the leaf <Text> that
  // actually carries the colour, not an ancestor Box.
  const deeper = findElementWithText(node.props.children, needle)

  if (deeper) {
    return deeper
  }

  return textContent(node).includes(needle) ? node : null
}

const baseProps = {
  bgCount: 0,
  busy: false,
  cols: 100,
  cwdLabel: '~/repo',
  liveSessionCount: 0,
  model: 'opus-4.8',
  sessionStartedAt: null,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, total: 50_000 },
  voiceLabel: ''
}

describe('StatusRule session title', () => {
  it('pins the named session at the far-right edge instead of the cwd label', () => {
    const element = StatusRule({
      ...baseProps,
      sessionTitle: 'weekly-digest'
    })

    const rendered = textContent(element)
    const title = findElementWithText(element, 'weekly-digest')

    expect(rendered).toContain('weekly-digest')
    expect(rendered).not.toContain('~/repo')
    // Regression for issue #82465: a raw, full-saturation accent-hue
    // background (e.g. #FFBF00 on DARK_SEEDS) paired with statusFg (a
    // near-white tone never designed to sit on it) rendered at roughly a
    // 1.5-2:1 contrast ratio -- unreadable. No background fill at all;
    // the accent color goes on the text instead, matching the theme's
    // own convention that a raw accent hue is never used as a solid
    // fill elsewhere (fills are always softened, e.g. activeRow).
    expect(title?.props.backgroundColor).toBeUndefined()
    expect(title?.props.color).toBe(DEFAULT_THEME.color.accent)
  })
})

describe('StatusRule background-subagent indicator', () => {
  it('renders ⛓ N on a wide terminal when subagents are running', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, active_subagents: 3 }
    })

    expect(textContent(element)).toContain('⛓ 3')
  })

  it('omits the segment when no subagents are running', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, active_subagents: 0 }
    })

    expect(textContent(element)).not.toContain('⛓')
  })

  it('omits the segment when the field is absent', () => {
    const element = StatusRule({ ...baseProps })

    expect(textContent(element)).not.toContain('⛓')
  })

  it('spells out the auto-resume hint when idle with subagents in flight', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, active_subagents: 1 }
    })

    expect(textContent(element)).toContain('resumes when subagent finishes')
  })

  it('pluralizes the resume hint for multiple in-flight subagents', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, active_subagents: 3 }
    })

    expect(textContent(element)).toContain('resumes when 3 subagents finish')
  })

  it('hides the resume hint mid-turn (a busy turn owns the indicator)', () => {
    const element = StatusRule({
      ...baseProps,
      busy: true,
      turnStartedAt: Date.now(),
      usage: { ...baseProps.usage, active_subagents: 2 }
    })

    expect(textContent(element)).not.toContain('resumes when')
  })

  it('omits the resume hint when no subagents are running', () => {
    const element = StatusRule({ ...baseProps })

    expect(textContent(element)).not.toContain('resumes when')
  })

  it('drops the subagent segment before the bg segment on a narrow terminal', () => {
    // cols=44 is below the subagents breakpoint (92) but the bg breakpoint
    // (88) too — both gone. Assert the lower-priority subagent indicator is
    // not shown when space is tight even with a live count.
    const element = StatusRule({
      ...baseProps,
      cols: 44,
      bgCount: 1,
      usage: { ...baseProps.usage, active_subagents: 2 }
    })

    expect(textContent(element)).not.toContain('⛓')
  })
})

describe('StatusRule session count click target', () => {
  it('makes the live session count itself clickable', () => {
    const openSwitcher = vi.fn()

    const element = StatusRule({
      bgCount: 0,
      busy: false,
      cols: 100,
      cwdLabel: '~/repo',
      liveSessionCount: 1,
      model: 'kimi-k2.6',
      onSessionCountClick: openSwitcher,
      sessionStartedAt: null,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { total: 0 },
      voiceLabel: ''
    })

    const clickableSessionCount = findClickableWithText(element, '1 session')

    expect(clickableSessionCount).not.toBeNull()
    clickableSessionCount!.props.onClick({ stopImmediatePropagation: vi.fn() })
    expect(openSwitcher).toHaveBeenCalledOnce()
  })

  it('keeps status + model and drops the low-value tail on a narrow terminal', () => {
    const element = StatusRule({
      bgCount: 0,
      busy: false,
      cols: 44,
      cwdLabel: '~/src/hermes-agent/apps/desktop (bb/tui-statusbar-responsive)',
      liveSessionCount: 3,
      model: 'opus-4.8',
      onSessionCountClick: vi.fn(),
      sessionStartedAt: Date.now() - 60_000,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: {
        calls: 0,
        context_max: 200_000,
        context_percent: 25,
        context_used: 50_000,
        input: 0,
        output: 0,
        total: 50_000
      },
      voiceLabel: 'voice off'
    })

    const rendered = textContent(element)

    // Must-keep essentials survive intact …
    expect(rendered).toContain('ready')
    expect(rendered).toContain('opus 4.8')
    // … while the low-value tail (session count) is dropped, not truncated.
    expect(rendered).not.toContain('3 sessions')
  })
})

describe('StatusRule credits notice render priority', () => {
  it('replaces the idle status with the notice text and keeps model + context', () => {
    const element = StatusRule({
      ...baseProps,
      notice: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ credits exhausted' }
    })

    const rendered = textContent(element)

    // Notice replaces the status verb slot …
    expect(rendered).toContain('✕ credits exhausted')
    expect(rendered).not.toContain('ready')
    // … but model + context stay visible.
    expect(rendered).toContain('opus 4.8')
    expect(rendered).toContain('50k')
  })

  it('busy wins: the FaceTicker shows, the notice is hidden mid-turn', () => {
    const element = StatusRule({
      ...baseProps,
      busy: true,
      notice: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' },
      turnStartedAt: Date.now()
    })

    const rendered = textContent(element)

    // Notice must NOT render while busy.
    expect(rendered).not.toContain('⚠ 90% used')
    // Model still visible.
    expect(rendered).toContain('opus 4.8')
  })

  it('colours the notice by level (error → theme error, success → statusGood)', () => {
    const errEl = StatusRule({
      ...baseProps,
      notice: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ exhausted' }
    })

    const errText = findElementWithText(errEl, '✕ exhausted')
    expect(errText?.props.color).toBe(DEFAULT_THEME.color.error)

    const okEl = StatusRule({
      ...baseProps,
      notice: { key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ restored', ttl_ms: 8000 }
    })

    const okText = findElementWithText(okEl, '✓ restored')
    expect(okText?.props.color).toBe(DEFAULT_THEME.color.statusGood)
  })

  it('does NOT add a glyph — the notice text is rendered verbatim', () => {
    const element = StatusRule({
      ...baseProps,
      notice: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' }
    })

    const noticeText = findElementWithText(element, '90% used')

    // The leaf carries exactly the policy text — no extra prepended glyph.
    expect(noticeText?.props.children).toBe('⚠ 90% used')
  })

  it('the notice text is the shrinkable element (flexShrink=1 + truncate-end) so a long notice ellipsizes', () => {
    const longText = '⚠ ' + 'x'.repeat(200)

    const element = StatusRule({
      ...baseProps,
      cols: 50,
      notice: { key: 'credits.90', kind: 'sticky', level: 'warn', text: longText }
    })

    // The leaf <Text> truncates rather than wrapping/clipping the pinned tail.
    const noticeText = findElementWithText(element, 'xxxxx')
    expect(noticeText?.props.wrap).toBe('truncate-end')

    // Its container box yields first (flexShrink=1) so model stays visible.
    const findShrinkBoxContaining = (node: ReactNodeLike): React.ReactElement | null => {
      if (!React.isValidElement(node)) {
        if (Array.isArray(node)) {
          for (const c of node) {
            const f = findShrinkBoxContaining(c)

            if (f) {
              return f
            }
          }
        }

        return null
      }

      if (node.props.flexShrink === 1 && textContent(node).includes('xxxxx') && node.type !== StatusRule) {
        // Prefer the closest shrink box that wraps the notice text.
        const deeper = findShrinkBoxContaining(node.props.children)

        return deeper ?? node
      }

      return findShrinkBoxContaining(node.props.children)
    }

    const shrinkBox = findShrinkBoxContaining(element)
    expect(shrinkBox).not.toBeNull()

    // Model survives on a narrow terminal because the notice yields.
    expect(textContent(element)).toContain('opus 4.8')
  })
})

describe('StatusRule battery indicator', () => {
  it('renders the battery label with a battery glyph on AC-off', () => {
    const element = StatusRule({
      ...baseProps,
      battery: { available: true, category: 'good', percent: 82, plugged: false }
    })

    expect(textContent(element)).toContain('🔋 82%')
  })

  it('uses a bolt glyph while charging', () => {
    const element = StatusRule({
      ...baseProps,
      battery: { available: true, category: 'good', percent: 82, plugged: true }
    })

    expect(textContent(element)).toContain('⚡ 82%')
  })

  it('colours the read-out by category (critical → theme statusCritical)', () => {
    const element = StatusRule({
      ...baseProps,
      battery: { available: true, category: 'critical', percent: 7, plugged: false }
    })

    const leaf = findElementWithText(element, '7%')
    expect(leaf?.props.color).toBe(DEFAULT_THEME.color.statusCritical)
  })

  it('omits the segment when battery is null', () => {
    const element = StatusRule({ ...baseProps, battery: null })

    expect(textContent(element)).not.toContain('%🔋')
    expect(textContent(element)).not.toContain('🔋')
  })

  it('omits the segment when no battery is available (desktop/server)', () => {
    const element = StatusRule({
      ...baseProps,
      battery: { available: false, category: 'dim', percent: null, plugged: null }
    })

    expect(textContent(element)).not.toContain('🔋')
  })
})

describe('StatusRule idle-since read-out', () => {
  // The IdleSince component uses hooks, so it can't be invoked outside a
  // renderer — assert on the element tree instead (same reason the duration
  // tests don't check SessionDuration's text).
  const findComponentByName = (node: ReactNodeLike, name: string): React.ReactElement | null => {
    if (node === null || node === undefined || typeof node === 'boolean') {
      return null
    }

    if (Array.isArray(node)) {
      for (const child of node) {
        const found = findComponentByName(child, name)

        if (found) {
          return found
        }
      }

      return null
    }

    if (!React.isValidElement(node)) {
      return null
    }

    if (typeof node.type === 'function' && node.type.name === name) {
      return node
    }

    return findComponentByName(node.props.children, name)
  }

  it('shows time since the last final agent response when idle', () => {
    const endedAt = Date.now() - 42_000

    const element = StatusRule({
      ...baseProps,
      lastTurnEndedAt: endedAt,
      sessionStartedAt: Date.now() - 60_000
    })

    const idle = findComponentByName(element, 'IdleSince')

    expect(idle).not.toBeNull()
    expect(idle!.props.endedAt).toBe(endedAt)
  })

  it('is hidden while a turn is busy', () => {
    const element = StatusRule({
      ...baseProps,
      busy: true,
      lastTurnEndedAt: Date.now() - 42_000,
      turnStartedAt: Date.now()
    })

    expect(findComponentByName(element, 'IdleSince')).toBeNull()
  })

  it('is hidden before the first turn completes', () => {
    const element = StatusRule({
      ...baseProps,
      lastTurnEndedAt: null,
      sessionStartedAt: Date.now() - 60_000
    })

    expect(findComponentByName(element, 'IdleSince')).toBeNull()
  })
})

describe('StatusRule perf read-outs (cache hit / latency / tps)', () => {
  const perfUsage = {
    ...baseProps.usage,
    avg_latency_s: 3.2,
    avg_tps: 50.4,
    cache_hit_pct: 87,
    calls: 4,
    input: 1000,
    output: 500
  }

  it('renders all three segments on a wide terminal', () => {
    const element = StatusRule({ ...baseProps, cols: 160, usage: perfUsage })
    const rendered = textContent(element)

    expect(rendered).toContain('◎ 87%')
    expect(rendered).toContain('◷ 3.2s')
    expect(rendered).toContain('↑ 50 t/s')
  })

  it('self-hides when the server omits the keys', () => {
    const element = StatusRule({ ...baseProps, cols: 160 })
    const rendered = textContent(element)

    expect(rendered).not.toContain('◎')
    expect(rendered).not.toContain('◷')
    expect(rendered).not.toContain('t/s')
  })

  it('honors the display.status_bar.fields visibility filter', () => {
    const element = StatusRule({
      ...baseProps,
      cols: 160,
      statusBarFields: new Set(['model', 'context_pct', 'cache_hit']),
      usage: perfUsage
    })

    const rendered = textContent(element)

    expect(rendered).toContain('◎ 87%')
    expect(rendered).not.toContain('◷')
    expect(rendered).not.toContain('t/s')
  })

  it('hides the session title badge when the fields filter omits title', () => {
    const element = StatusRule({
      ...baseProps,
      cols: 160,
      sessionTitle: 'weekly-digest',
      statusBarFields: new Set(['model', 'context_pct'])
    })

    expect(textContent(element)).not.toContain('weekly-digest')
  })
})
