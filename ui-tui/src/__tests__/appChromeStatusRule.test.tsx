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
  showCost: false,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, total: 50_000 },
  voiceLabel: ''
}

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
      showCost: false,
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
      showCost: true,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, cost_usd: 0.5, total: 50_000 },
      voiceLabel: 'voice off'
    })

    const rendered = textContent(element)

    // Must-keep essentials survive intact …
    expect(rendered).toContain('ready')
    expect(rendered).toContain('opus 4.8')
    // … while the low-value tail (session count, cost) is dropped, not truncated.
    expect(rendered).not.toContain('3 sessions')
    expect(rendered).not.toContain('$0.5000')
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
            if (f) return f
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

describe('StatusRule runtime status segments', () => {
  it('renders runtime tool, skill, and task on wide terminals', () => {
    const element = StatusRule({
      ...baseProps,
      cols: 180,
      usage: {
        ...baseProps.usage,
        runtime: {
          phase: 'implement',
          run_mode: 'implement',
          target: 'tui',
          main_agent: 'executor',
          recent_tool: { name: 'terminal', status: 'ok', duration_ms: 17 },
          recent_skill: { name: 'hermes-agent', event: 'view' },
          active_subagent: { id: 'sa-1', label: 'verifier', status: 'running', last_tool: 'read_file', tool_count: 1 },
          background_tasks: { running: 2 },
          task: { total: 4, completed: 2, in_progress: 0, pending: 2, cancelled: 0 },
          wait: { reason: 'none', since: null }
        }
      }
    })

    const rendered = textContent(element)

    expect(rendered).toContain('tool:terminal✓')
    expect(rendered).toContain('skill:hermes-agent')
    expect(rendered).toContain('sub:verifier')
    expect(rendered).toContain('task:2/4')
    expect(rendered).toContain('bg:2')
  })

  it('renders distinct phase when activity phase differs from run mode', () => {
    const element = StatusRule({
      ...baseProps,
      cols: 180,
      usage: {
        ...baseProps.usage,
        runtime: {
          phase: 'thinking',
          run_mode: 'agent',
          target: '',
          main_agent: 'main',
          wait: { reason: 'model', since: 123 }
        }
      }
    })

    const rendered = textContent(element)

    expect(rendered).toContain('run:agent')
    expect(rendered).toContain('phase:thinking')
    expect(rendered).toContain('wait:model')
  })

  it('prioritizes run and phase before runtime tail on constrained terminals', () => {
    const element = StatusRule({
      ...baseProps,
      cols: 96,
      usage: {
        ...baseProps.usage,
        runtime: {
          phase: 'waiting',
          run_mode: 'implement',
          target: 'claude-code',
          main_agent: 'executor',
          recent_tool: { name: 'terminal', status: 'ok' },
          recent_skill: { name: 'hermes-agent', event: 'view' },
          active_subagent: { id: 'sa-1', label: 'verifier', status: 'running', last_tool: 'read_file', tool_count: 1 },
          background_tasks: { running: 2 },
          task: { total: 8, completed: 3 },
          wait: { reason: 'approval', since: 123 }
        }
      }
    })

    const rendered = textContent(element)

    expect(rendered).toContain('run:implement')
    expect(rendered).toContain('phase:waiting')
    expect(rendered).not.toContain('tool:terminal')
    expect(rendered).not.toContain('sub:verifier')
    expect(rendered).not.toContain('skill:hermes-agent')
    expect(rendered).not.toContain('task:3/8')
    expect(rendered).not.toContain('bg:2')
    expect(rendered).not.toContain('wait:approval')
  })

  it('renders error tool status with a failure glyph', () => {
    const element = StatusRule({
      ...baseProps,
      cols: 180,
      usage: {
        ...baseProps.usage,
        runtime: {
          phase: 'implement',
          run_mode: 'implement',
          target: '',
          main_agent: 'main',
          recent_tool: { name: 'terminal', status: 'error', error_message: 'boom' },
          wait: { reason: 'none', since: null }
        }
      }
    })

    expect(textContent(element)).toContain('tool:terminal✗')
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
