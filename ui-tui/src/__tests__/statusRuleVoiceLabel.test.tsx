import React from 'react'
import { describe, expect, it } from 'vitest'

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

  if (React.isValidElement<{ children?: React.ReactNode }>(node)) {
    return textContent(node.props.children)
  }

  return ''
}

const statusRuleText = (voiceLabel: string): string =>
  textContent(
    StatusRule({
      bgCount: 0,
      busy: false,
      cols: 80,
      cwdLabel: '~/repo',
      fields: ['voice'],
      liveSessionCount: 1,
      model: 'gpt-5.5',
      sessionStartedAt: null,
      showCost: false,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { calls: 0, input: 0, output: 0, total: 0 },
      voiceLabel
    })
  )

describe('StatusRule voice label rendering', () => {
  it('omits the voice segment when voiceLabel is empty', () => {
    const text = statusRuleText('')

    expect(text).not.toContain('voice')
    expect(text).not.toContain('│')
  })

  it('renders non-empty voice indicators', () => {
    expect(statusRuleText('voice on')).toContain('voice on')
    expect(statusRuleText('● REC')).toContain('● REC')
  })
})
