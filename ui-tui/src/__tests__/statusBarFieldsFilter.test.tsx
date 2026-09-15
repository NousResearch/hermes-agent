/**
 * Regression for the status-bar field filter never sticking.
 *
 * Two independent defects made `display.status_bar.fields` look ignored:
 *
 *  1. `undefined` (config not hydrated yet) and `null` (hydrated, not
 *     customized) both meant "show everything", and the boot state was
 *     `null` — so the first render was indistinguishable from a configured
 *     filter, and a failed/racing config.get reset the filter back to
 *     "show all" instead of leaving it alone.
 *  2. `context_detail` (used/max) and `context_pct` (% + bar) shared one
 *     OR'd gate, so asking for the percentage alone still rendered the
 *     used/max detail. The idle clock and the live-session counter had no
 *     gate at all.
 *
 * These tests assert the contract between the config list and what renders.
 * Text extraction and the call style mirror appChromeStatusRule.test.tsx.
 */
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

const baseProps = {
  bgCount: 0,
  busy: false,
  cols: 200,
  cwdLabel: '~/repo',
  liveSessionCount: 1,
  model: 'deepseek flash',
  sessionStartedAt: null as null | number,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null as null | number,
  usage: {
    context_max: 1_000_000,
    context_percent: 7,
    context_used: 68_800,
    avg_tps: 95,
    total: 68_800
  } as any,
  voiceLabel: ''
}

/** Render the rule for a given filter and flatten it to visible text. */
const renderRule = (fields: null | ReadonlySet<string> | undefined) =>
  textContent(StatusRule({ ...baseProps, statusBarFields: fields } as any))

describe('display.status_bar.fields', () => {
  it('undefined (not hydrated) shows the default set, never a blank bar', () => {
    const text = renderRule(undefined)

    // A pre-hydration render must not be treated as an empty filter.
    expect(text).toContain('68.8k/1M')
    expect(text).toContain('deepseek flash')
  })

  it('null (hydrated, uncustomized) shows the default set', () => {
    const text = renderRule(null)

    expect(text).toContain('68.8k/1M')
    expect(text).toContain('deepseek flash')
  })

  it('context_pct alone renders the percentage and never the used/max detail', () => {
    const text = renderRule(new Set(['model', 'context_pct']))

    expect(text).toContain('7%')
    // The defect: the OR'd gate rendered the detail anyway.
    expect(text).not.toContain('68.8k/1M')
  })

  it('context_detail alone renders used/max and never the percentage read-out', () => {
    const text = renderRule(new Set(['model', 'context_detail']))

    expect(text).toContain('68.8k/1M')
    expect(text).not.toContain('7%')
  })

  it('omits unlisted segments that previously had no gate at all', () => {
    // `duration` and `sessions` are both absent from this list, and both
    // used to render regardless of the filter.
    const text = renderRule(new Set(['model', 'context_pct']))

    expect(text).not.toContain('session')
    expect(text).not.toContain('t/s')
  })

  it('shows only the gated segments that are both listed and populated', () => {
    const text = renderRule(new Set(['model', 'context_pct', 'tps', 'sessions']))

    expect(text).toContain('deepseek flash')
    expect(text).toContain('7%')
    expect(text).toContain('95 t/s')
    expect(text).toContain('1 session')
    expect(text).not.toContain('68.8k/1M')
  })

  it('hides the idle clock when duration is filtered out', () => {
    // The idle clock is driven by a live timer, so only its gate is
    // assertable in a static render: unlisted `duration` must not surface.
    const text = textContent(
      StatusRule({ ...baseProps, lastTurnEndedAt: Date.now() - 28_000, statusBarFields: new Set(['model']) } as any)
    )

    expect(text).not.toContain('28s')
  })
})
