import React from 'react'
import { describe, expect, it } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { ZERO } from '../domain/usage.js'
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

describe('StatusRule context display', () => {
  it('keeps the token count and omits the visual percentage bar', () => {
    const text = textContent(
      StatusRule({
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
        usage: {
          ...ZERO,
          context_max: 20_000,
          context_percent: 50,
          context_used: 10_000,
          total: 12_000
        },
        voiceLabel: ''
      })
    )

    expect(text).toContain('10k/20k')
    expect(text).toContain('ready │ gpt 5.5 │ 10k/20k ─ ~/repo')
    expect(text).not.toContain('[█████░░░░░]')
    expect(text).not.toContain('50%')
    expect(text).not.toContain('│  │')
  })
})
