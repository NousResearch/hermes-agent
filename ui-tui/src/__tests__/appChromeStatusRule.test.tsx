import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Usage } from '../types.js'

type ReactNodeLike = React.ReactNode

const emptyUsage: Usage = { calls: 0, input: 0, output: 0, total: 0 }

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

const findElementsWithExactText = (node: ReactNodeLike, exact: string): React.ReactElement<Record<string, any>>[] => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return []
  }

  if (Array.isArray(node)) {
    return node.flatMap(child => findElementsWithExactText(child, exact))
  }

  if (!React.isValidElement(node)) {
    return []
  }

  const here = node.props.children === exact ? [node] : []

  return [...here, ...findElementsWithExactText(node.props.children, exact)]
}

describe('StatusRule compact footer', () => {
  it('keeps the footer to model, context, and session time only', () => {
    const openSwitcher = vi.fn()
    const element = StatusRule({
      bgCount: 7,
      busy: true,
      cols: 120,
      cwdLabel: '~/repo',
      liveSessionCount: 2,
      model: 'openai/gpt-5.5',
      modelReasoningEffort: 'xhigh',
      onSessionCountClick: openSwitcher,
      sessionStartedAt: Date.now(),
      showCost: true,
      status: 'running',
      statusColor: DEFAULT_THEME.color.warn,
      t: DEFAULT_THEME,
      turnStartedAt: Date.now(),
      usage: {
        ...emptyUsage,
        context_max: 272000,
        context_percent: 28,
        context_used: 74900,
        cost_usd: 1.23,
        compressions: 3
      },
      voiceLabel: 'voice on'
    })

    const text = textContent(element)

    expect(text).toContain('gpt 5.5 xhigh')
    expect(text).toContain('Context 74.9k/272k 28% [███░░░░░░░]')
    const contextSeparators = findElementsWithExactText(element, ' │ ')
    expect(contextSeparators.length).toBeGreaterThanOrEqual(2)
    expect(contextSeparators.every(node => node.props.color === DEFAULT_THEME.color.muted)).toBe(true)
    expect(text).not.toContain('Ctrl+C interrupt')
    expect(text).not.toContain('voice')
    expect(text).not.toContain('sessions')
    expect(text).not.toContain('bg')
    expect(text).not.toContain('cmp')
    expect(text).not.toContain('$')
    expect(findClickableWithText(element, '2 sessions')).toBeNull()
    expect(openSwitcher).not.toHaveBeenCalled()
  })
})
