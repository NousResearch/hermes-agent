import React from 'react'
import { describe, expect, it } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Usage } from '../types.js'

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
    return textContent((node.props as { children?: ReactNodeLike }).children)
  }

  return ''
}

// Collect every (color, text) pair so a test can assert the colour applied to
// the segment carrying the credit label.
const collectColored = (node: ReactNodeLike, out: { color: unknown; text: string }[] = []) => {
  if (Array.isArray(node)) {
    node.forEach(child => collectColored(child, out))

    return out
  }

  if (React.isValidElement(node)) {
    const props = node.props as { children?: ReactNodeLike; color?: unknown }
    if (props.color) {
      out.push({ color: props.color, text: textContent(props.children) })
    }

    collectColored(props.children, out)
  }

  return out
}

const baseUsage: Usage = { calls: 0, context_max: 200_000, context_percent: 25, context_used: 50_000, input: 0, output: 0, total: 50_000 }

const baseProps = {
  bgCount: 0,
  busy: false,
  cols: 200,
  cwdLabel: '~/repo',
  liveSessionCount: 0,
  model: 'opus-4.8',
  sessionStartedAt: null,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: baseUsage,
  voiceLabel: ''
}

describe('StatusRule OpenRouter credits', () => {
  it('renders the 💳 credit label when a balance is present', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseUsage, or_credits_balance: 42.5, or_credits_label: '$42.50' }
    } as never)

    expect(textContent(element)).toContain('💳 $42.50')
  })

  it('hides the credit segment when no balance is present', () => {
    const element = StatusRule(baseProps as never)

    expect(textContent(element)).not.toContain('💳')
  })

  it('colours a critical (<=$1) balance with the critical status colour', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseUsage, or_credits_balance: 0.5, or_credits_label: '$0.50' }
    } as never)

    const segment = collectColored(element).find(entry => entry.text.includes('💳 $0.50'))
    expect(segment?.color).toBe(DEFAULT_THEME.color.statusCritical)
  })

  it('colours a healthy (>$50) balance with the good status colour', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseUsage, or_credits_balance: 120, or_credits_label: '$120.00' }
    } as never)

    const segment = collectColored(element).find(entry => entry.text.includes('💳 $120.00'))
    expect(segment?.color).toBe(DEFAULT_THEME.color.statusGood)
  })
})
