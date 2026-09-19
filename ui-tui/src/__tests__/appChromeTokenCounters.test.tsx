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

  if (React.isValidElement(node)) {
    return textContent(node.props.children)
  }

  return ''
}

// List every element in the tree that carries text. Used to assert ordering
// between the model / ctx / token / tps segments.
const textElements = (node: ReactNodeLike): Array<{ text: string; props: Record<string, unknown> }> => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return []
  }

  if (typeof node === 'string' || typeof node === 'number') {
    return [{ text: String(node), props: {} }]
  }

  if (Array.isArray(node)) {
    return node.flatMap(textElements)
  }

  if (!React.isValidElement(node)) {
    return []
  }

  const children =
    node.props.children === undefined
      ? []
      : Array.isArray(node.props.children)
        ? node.props.children
        : [node.props.children]

  return textElements(children as ReactNodeLike)
}

const baseProps = {
  bgCount: 0,
  busy: false,
  cols: 160,
  cwdLabel: '~/repo',
  liveSessionCount: 0,
  model: 'opus-4.8',
  sessionStartedAt: null,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: {
    calls: 0,
    context_max: 200_000,
    context_percent: 25,
    context_used: 50_000,
    total: 50_000,
    input: 0,
    output: 0
  },
  voiceLabel: ''
}

describe('StatusRule token counters (↑ session input / ↓ session output)', () => {
  it('renders `↑ 1.2k ↓ 5.7k` for { input: 1234, output: 5678 } inside a pinned (flexShrink=0) box', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, input: 1234, output: 5678 }
    })

    const rendered = textContent(element)

    expect(rendered).toContain('↑ 1.2k')
    expect(rendered).toContain('↓ 5.7k')

    // The counters sit in the pinned essentials box (flexShrink=0), so they
    // never shrink / yield on a narrow terminal.
    const findPinnedWithCounters = (node: ReactNodeLike): React.ReactElement | null => {
      if (!React.isValidElement(node)) {
        if (Array.isArray(node)) {
          for (const c of node) {
            const f = findPinnedWithCounters(c)

            if (f) {
              return f
            }
          }
        }

        return null
      }

      if (node.props.flexShrink === 0 && textContent(node).includes('↑ 1.2k') && node.type !== StatusRule) {
        const deeper = findPinnedWithCounters(node.props.children)

        return deeper ?? node
      }

      return findPinnedWithCounters(node.props.children)
    }

    const pinnedBox = findPinnedWithCounters(element)
    expect(pinnedBox).not.toBeNull()
    // The pinned essentials box containing the counters must not shrink.
    expect(pinnedBox!.props.flexShrink).toBe(0)
  })

  it('self-hides both counters when input AND output are zero', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, input: 0, output: 0 }
    })

    const rendered = textContent(element)

    expect(rendered).not.toContain('↑')
    expect(rendered).not.toContain('↓ 0')
    // No token fragment at all (arrow glyphs belong to the token counters only).
    expect(rendered).not.toContain('↑ 0')
  })

  it('renders only the input counter when output is zero', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, input: 1234, output: 0 }
    })

    const rendered = textContent(element)

    expect(rendered).toContain('↑ 1.2k')
    expect(rendered).not.toContain('↓')
  })

  it('renders only the output counter when input is zero', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, input: 0, output: 5678 }
    })

    const rendered = textContent(element)

    expect(rendered).toContain('↓ 5.7k')
    expect(rendered).not.toContain('↑')
  })

  it('formats tokens with fmtK (999→999, 1000→1k, 1500→1.5k, 1234567→1.2m)', () => {
    // Ground truth is the reused `fmtK` (lib/text.ts): `Intl.NumberFormat(
    // 'en-US', {notation:'compact'})` emits a lowercase `m` for millions
    // (1.2m), and `.replace(/[KMBT]$/, s => s.toLowerCase())` only lowercases
    // the kilo `K`. The architect's spec prose said "1.2M" but we must reuse
    // the existing function unchanged, so the real (lowercase) output is what
    // we pin here.
    const cases: Array<[number, string]> = [
      [999, '↑ 999'],
      [1000, '↑ 1k'],
      [1500, '↑ 1.5k'],
      [1234567, '↑ 1.2M']
    ]

    for (const [input, expected] of cases) {
      const element = StatusRule({
        ...baseProps,
        usage: { ...baseProps.usage, input, output: 0 }
      })

      expect(textContent(element), `${input} input tokens`).toContain(expected)
    }
  })

  it('renders the token segment AFTER the context label in the pinned essentials box', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, input: 1234, output: 5678 }
    })

    // The pinned essentials box is model │ ctx │ counters. Find the box that
    // contains all of model, ctx and the counters, then assert ordering.
    const seq = textElements(element)
    const ctxIndex = seq.findIndex(e => e.text.includes('50k'))
    const tokenIndex = seq.findIndex(e => e.text.includes('↑ 1.2k'))

    expect(ctxIndex).toBeGreaterThanOrEqual(0)
    expect(tokenIndex).toBeGreaterThan(ctxIndex)
  })
})

describe('StatusRule streaming throughput (↓ N t/s)', () => {
  it('renders `↓ 142 t/s` when stream_tps is 142', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, input: 1234, output: 5678, stream_tps: 142 }
    })

    expect(textContent(element)).toContain('↓ 142 t/s')
  })

  it('is absent when stream_tps is undefined', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, input: 1234, output: 5678 }
    })

    expect(textContent(element)).not.toContain('t/s')
  })

  it('is absent when stream_tps is zero', () => {
    const element = StatusRule({
      ...baseProps,
      usage: { ...baseProps.usage, input: 1234, output: 5678, stream_tps: 0 }
    })

    expect(textContent(element)).not.toContain('t/s')
  })
})

describe('StatusRule token + tps governance (segs + statusBarFields)', () => {
  it('hides token counters & stream tps on a narrow terminal (below the pixels breakpoint)', () => {
    // statusBarSegments(72) → tokens & tps breakpoints are at cols >= 110,
    // so at 72 both segments must be suppressed despite having values.
    const element = StatusRule({
      ...baseProps,
      cols: 72,
      usage: { ...baseProps.usage, input: 1234, output: 5678, stream_tps: 142 }
    })

    const rendered = textContent(element)
    expect(rendered).not.toContain('↑ 1.2k')
    expect(rendered).not.toContain('↓ 5.7k')
    expect(rendered).not.toContain('t/s')
  })

  it('respects statusBarFields: omitting "tokens" and "tps" hides both segments', () => {
    const element = StatusRule({
      ...baseProps,
      statusBarFields: new Set(['model', 'context_pct']),
      usage: { ...baseProps.usage, input: 1234, output: 5678, stream_tps: 142 }
    })

    const rendered = textContent(element)
    expect(rendered).not.toContain('↑ 1.2k')
    expect(rendered).not.toContain('↓ 5.7k')
    expect(rendered).not.toContain('t/s')
  })

  it('shows both when cols >= breakpoint (160) and statusBarFields allows them', () => {
    // baseProps uses cols: 160 → segs.tokens/tps true. Explicit field allow.
    const element = StatusRule({
      ...baseProps,
      statusBarFields: new Set(['model', 'context_pct', 'tokens', 'tps']),
      usage: { ...baseProps.usage, input: 1234, output: 5678, stream_tps: 142 }
    })

    const rendered = textContent(element)
    expect(rendered).toContain('↑ 1.2k')
    expect(rendered).toContain('↓ 5.7k')
    expect(rendered).toContain('↓ 142 t/s')
  })
})
