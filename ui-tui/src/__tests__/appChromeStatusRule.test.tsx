import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { ZERO } from '../domain/usage.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

type ElementProps = {
  children?: React.ReactNode
  onClick?: unknown
}
type StatusRulePropsForTest = Parameters<typeof StatusRule>[0]

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

  if (React.isValidElement<ElementProps>(node)) {
    const element = node as React.ReactElement<ElementProps>

    return textContent(element.props.children)
  }

  return ''
}

const findClickableWithText = (node: ReactNodeLike, needle: string): React.ReactElement<ElementProps> | null => {
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

  if (!React.isValidElement<ElementProps>(node)) {
    return null
  }

  const element = node as React.ReactElement<ElementProps>

  if (typeof element.props.onClick === 'function' && textContent(element).includes(needle)) {
    return element
  }

  return findClickableWithText(element.props.children, needle)
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

const baseProps: StatusRulePropsForTest = {
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
  usage: { ...ZERO, context_max: 200_000, context_percent: 25, context_used: 50_000, total: 50_000 },
  voiceLabel: ''
}

const renderStatusRuleText = (overrides: Partial<StatusRulePropsForTest>) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 120, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(React.createElement(StatusRule, { ...baseProps, ...overrides }), {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

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
    // cols=44 is below the subagents breakpoint (92) and the bg breakpoint
    // (88), so both drop. Assert the lower-priority subagent indicator is
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

describe('StatusRule busy indicator', () => {
  it('renders the static working label with spinner, elapsed duration, model, and context', () => {
    const text = renderStatusRuleText({
      busy: true,
      indicatorStyle: 'ascii',
      turnStartedAt: Date.now() - 2500
    })

    expect(text).toContain('working')
    expect(text).toMatch(/ · \d+s/)
    expect(text).toContain('opus 4.8')
    expect(text).toContain('50k')
    expect(text).not.toMatch(/pondering|contemplating|reasoning|synthesizing/)
  })

  it('keeps unicode style textless while still showing spinner, duration, model, and context', () => {
    const text = renderStatusRuleText({
      busy: true,
      indicatorStyle: 'unicode',
      turnStartedAt: Date.now() - 2500
    })

    expect(text).not.toContain('working')
    expect(text).toMatch(/ · \d+s/)
    expect(text).toContain('opus 4.8')
    expect(text).toContain('50k')
  })
})


describe('StatusRule live session count', () => {
  it('hides the live session count from the status bar', () => {
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
      usage: ZERO,
      voiceLabel: ''
    })

    const text = textContent(element)
    const clickableSessionCount = findClickableWithText(element, '1 session')

    expect(text).not.toContain('1 session')
    expect(clickableSessionCount).toBeNull()
    expect(openSwitcher).not.toHaveBeenCalled()
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
      usage: { ...ZERO, context_max: 200_000, context_percent: 25, context_used: 50_000, cost_usd: 0.5, total: 50_000 },
      voiceLabel: 'voice off'
    })

    const rendered = textContent(element)

    // Must-keep essentials survive intact …
    expect(rendered).toContain('ready')
    expect(rendered).toContain('opus 4.8')
    // … while the low-value tail (session count) is dropped, not truncated.
    expect(rendered).not.toContain('3 sessions')
  })

  it('filters status rule fields from config', () => {
    const element = StatusRule({
      bgCount: 2,
      busy: false,
      cols: 120,
      cwdLabel: '~/repo',
      fields: ['status', 'model', 'context', 'delegation', 'background', 'cwd'],
      liveSessionCount: 3,
      model: 'gpt-5.5',
      sessionStartedAt: Date.now() - 60_000,
      showCost: true,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: {
        ...ZERO,
        compressions: 2,
        context_max: 20_000,
        context_percent: 50,
        context_used: 10_000,
        cost_usd: 0.1234,
        total: 12_000
      },
      voiceLabel: '◯ voice'
    })

    const text = textContent(element)

    expect(text).toContain('ready')
    expect(text).toContain('gpt 5.5')
    expect(text).toContain('10k/20k')
    expect(text).not.toContain('[█████░░░░░] 50%')
    expect(text).not.toContain('50%')
    expect(text).toContain('2 bg')
    expect(text).toContain('~/repo')
    expect(text).not.toContain('3 sessions')
    expect(text).not.toContain('cmp 2')
    expect(text).not.toContain('$0.1234')
    expect(text).not.toContain('voice')
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
