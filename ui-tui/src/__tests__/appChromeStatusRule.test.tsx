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

  it('renders native quota when present in usage', () => {
    const quotaLabel = 'cdx 5h 12%\u21bb2h 7d 4%\u21bb5d'
    const element = StatusRule({
      bgCount: 0,
      busy: false,
      cols: 120,
      cwdLabel: '~/repo',
      liveSessionCount: 0,
      model: 'gpt-5.5',
      sessionStartedAt: null,
      showCost: false,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { calls: 0, input: 0, output: 0, total: 0, native_quota: quotaLabel },
      voiceLabel: ''
    })

    const allText = textContent(element)
    expect(allText).toContain(quotaLabel)
  })

  it('omits native quota when absent from usage', () => {
    const element = StatusRule({
      bgCount: 0,
      busy: false,
      cols: 120,
      cwdLabel: '~/repo',
      liveSessionCount: 0,
      model: 'gpt-5.5',
      sessionStartedAt: null,
      showCost: false,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { calls: 0, input: 0, output: 0, total: 0 },
      voiceLabel: ''
    })

    const allText = textContent(element)
    expect(allText).not.toContain('cdx')
    expect(allText).not.toContain('\u21bb')
  })
})
