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

const coloredTextSegments = (node: ReactNodeLike): { color?: string; text: string }[] => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return []
  }

  if (Array.isArray(node)) {
    return node.flatMap(coloredTextSegments)
  }

  if (!React.isValidElement(node)) {
    return []
  }

  const own =
    typeof node.props.color === 'string' ? [{ color: node.props.color, text: textContent(node.props.children) }] : []

  return [...own, ...coloredTextSegments(node.props.children)]
}

const colorForText = (node: ReactNodeLike, needle: string) =>
  coloredTextSegments(node).find(segment => segment.text.includes(needle))?.color

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
})

describe('StatusRule compact phone layout', () => {
  it('keeps model and context usage readable in narrow terminals', () => {
    const element = StatusRule({
      bgCount: 0,
      busy: false,
      cols: 58,
      cwdLabel: '~/Workspaces',
      liveSessionCount: 1,
      model: 'dflash',
      sessionStartedAt: null,
      showCost: false,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: {
        context_max: 262000,
        context_percent: 8,
        context_used: 20900,
        total: 20900
      },
      voiceLabel: ''
    })

    const content = textContent(element)

    expect(content).toContain('dflash')
    expect(content).toContain('ctx 20.9k/262k 8%')
    expect(content).toContain('1 session')
    expect(content).toContain('~/Workspaces')
    expect(colorForText(element, '- ready')).toBe(DEFAULT_THEME.color.statusGood)
    expect(colorForText(element, 'dflash')).toBe(DEFAULT_THEME.color.primary)
    expect(colorForText(element, 'ctx 20.9k/262k 8%')).toBe(DEFAULT_THEME.color.statusGood)
    expect(colorForText(element, '1 session')).toBe(DEFAULT_THEME.color.accent)
    expect(colorForText(element, '~/Workspaces')).toBe(DEFAULT_THEME.color.label)
  })

  it('colors compact context by low, moderate, and high usage', () => {
    const renderWithContext = (contextPercent: number) =>
      StatusRule({
        bgCount: 0,
        busy: false,
        cols: 58,
        cwdLabel: '~/Workspaces',
        liveSessionCount: 1,
        model: 'dflash',
        sessionStartedAt: null,
        showCost: false,
        status: 'ready',
        statusColor: DEFAULT_THEME.color.ok,
        t: DEFAULT_THEME,
        turnStartedAt: null,
        usage: {
          context_max: 100000,
          context_percent: contextPercent,
          context_used: contextPercent * 1000,
          total: contextPercent * 1000
        },
        voiceLabel: ''
      })

    expect(colorForText(renderWithContext(8), 'ctx 8k/100k 8%')).toBe(DEFAULT_THEME.color.statusGood)
    expect(colorForText(renderWithContext(60), 'ctx 60k/100k 60%')).toBe(DEFAULT_THEME.color.statusWarn)
    expect(colorForText(renderWithContext(85), 'ctx 85k/100k 85%')).toBe(DEFAULT_THEME.color.statusCritical)
  })

  it('does not spill compact busy status words across phone lines', () => {
    const now = new Date('2026-05-31T22:30:00Z').getTime()
    vi.useFakeTimers()
    vi.setSystemTime(now)

    try {
      const element = StatusRule({
        bgCount: 0,
        busy: true,
        cols: 58,
        cwdLabel: '~/Workspaces',
        liveSessionCount: 1,
        model: 'dflash',
        sessionStartedAt: now - 90_000,
        showCost: false,
        status: 'deliberating...',
        statusColor: DEFAULT_THEME.color.ok,
        t: DEFAULT_THEME,
        turnStartedAt: now - 45_000,
        usage: {
          context_estimated: true,
          context_max: 262000,
          context_percent: 8,
          context_used: 20900,
          total: 20900
        },
        voiceLabel: 'voice off'
      })

      const content = textContent(element)

      expect(content).toContain('- busy 45s | dflash | ctx ~20.9k/262k 8%')
      expect(content).toContain('dur 1m 30s | voice off | 1 session | ~/Workspaces')
      expect(content).not.toContain('deliberating')
      expect(content).not.toContain('model dfla')
    } finally {
      vi.useRealTimers()
    }
  })
})
