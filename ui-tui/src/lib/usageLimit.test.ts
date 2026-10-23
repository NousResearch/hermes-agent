import type { ModelOptionProvider } from '@hermes/shared/gateway-events'
import {
  blockingGauge,
  formatResetsIn,
  PLAN_LIMIT_WORDS_EN,
  planBlocks,
  planLimitLine,
  usageTag
} from '@hermes/shared/plan-usage'
import { describe, expect, it } from 'vitest'

import { usageLimitDialog } from './billingDialog.js'

const NOW = 1_000_000_000_000
const HOUR = 3_600_000

type Provider = Partial<ModelOptionProvider>

const window = (
  gauges: { label: string; left_fraction: number; resets_at_ms?: number; scope?: string }[]
): Provider => ({
  plan_usage: { gauges: gauges.map(g => ({ style: 'bar', ...g })), kind: 'usage_window' },
  usage_levels: { 'a/free:free': 'free', 'a/low': 'low', 'a/large': 'top_tier', 'a/zero': 'free' }
})

describe('blockingGauge / planBlocks (the one rule, agent/plan_limits.py twin)', () => {
  const dayFull = window([
    { label: 'day', left_fraction: 0, resets_at_ms: NOW + 3 * HOUR, scope: 'all' },
    { label: 'week', left_fraction: 0.5, resets_at_ms: NOW + 72 * HOUR, scope: 'all' },
    { label: 'top_tier', left_fraction: 0.4, resets_at_ms: NOW + 500 * HOUR, scope: 'top_tier' }
  ])

  const topFull = window([
    { label: 'day', left_fraction: 0.6, resets_at_ms: NOW + 3 * HOUR, scope: 'all' },
    { label: 'top_tier', left_fraction: 0, resets_at_ms: NOW + 50 * HOUR, scope: 'top_tier' }
  ])

  it('a full all-scope gauge blocks every paid model', () => {
    expect(blockingGauge(dayFull, 'a/low')?.label).toBe('day')
    expect(blockingGauge(dayFull, 'a/large')?.label).toBe('day')
    expect(planBlocks(dayFull, 'a/low')).toBe(true)
  })

  it('a full top-tier gauge blocks only top-tier models', () => {
    expect(planBlocks(topFull, 'a/large')).toBe(true)
    expect(planBlocks(topFull, 'a/low')).toBe(false)
  })

  it('free models are never used up (gateway word, :free id)', () => {
    expect(planBlocks(dayFull, 'a/zero')).toBe(false)
    expect(planBlocks(dayFull, 'other/x:free')).toBe(false)
  })

  it('of several full gauges, the one that frees up last', () => {
    const both = window([
      { label: 'day', left_fraction: 0, resets_at_ms: NOW + HOUR, scope: 'all' },
      { label: 'week', left_fraction: 0, resets_at_ms: NOW + 30 * HOUR, scope: 'all' }
    ])

    expect(blockingGauge(both, 'a/low')?.label).toBe('week')
  })

  it('non-window accounts are never blocked or tagged', () => {
    expect(planBlocks({ plan_usage: { kind: 'credits' } } as Provider, 'a/low')).toBe(false)
    expect(planBlocks(undefined, 'a/low')).toBe(false)
    expect(usageTag({ plan_usage: { kind: 'credits' } } as Provider, 'a/large')).toBeNull()
    expect(planLimitLine({ plan_usage: { kind: 'credits' } } as Provider, NOW)).toBe('')
  })

  const paying = (p: Provider, creditsPay: boolean | null): Provider => ({
    ...p,
    plan_usage: { ...p.plan_usage!, credits_pay: creditsPay }
  })

  it('pay as you go: rows stay usable, the limit line says paid models now use credits', () => {
    expect(planBlocks(paying(dayFull, true), 'a/low')).toBe(false)
    expect(planLimitLine(paying(dayFull, true), NOW)).toBe(
      'Daily limit reached · resets in 3h · paid models now use your credits (pay as you go)'
    )
    expect(planLimitLine(paying(topFull, true), NOW)).toBe(
      'Monthly large-model limit reached · resets in 2d 2h · Large models now use your credits (pay as you go)'
    )
  })

  it('credits will not pay: the limit line says what still works', () => {
    expect(planLimitLine(paying(dayFull, false), NOW)).toBe(
      'Daily limit reached · resets in 3h · only Free models until then'
    )
    expect(planLimitLine(paying(topFull, false), NOW)).toBe(
      'Monthly large-model limit reached · resets in 2d 2h · Large models paused until then'
    )
    expect(planLimitLine(paying(dayFull, null), NOW)).toBe('Daily limit reached · resets in 3h')
  })

  it('row tags: only Large and Free; Small / Medium rows carry none', () => {
    expect(usageTag(dayFull, 'a/large')).toBe('top_tier')
    expect(usageTag(dayFull, 'a/free:free')).toBe('free')
    expect(usageTag(dayFull, 'a/low')).toBeNull()
  })

  it('formats the reset compactly', () => {
    expect(formatResetsIn(NOW + 30_000, NOW)).toBe('soon')
    expect(formatResetsIn(NOW + 12 * 60_000, NOW)).toBe('12m')
    expect(formatResetsIn(NOW + 5 * HOUR, NOW)).toBe('5h')
    expect(formatResetsIn(NOW + 48 * HOUR, NOW)).toBe('2d')
    expect(formatResetsIn(null, NOW)).toBe('')
  })
})

describe('usageLimitDialog', () => {
  const limit = {
    credits: { action: 'add_credits' },
    gauge_id: 'day',
    limit_name: 'daily',
    message: "You've hit your daily limit · resets 15:56",
    scope: 'all',
    suggestions: [{ label: 'Model Free', model: 'example/model-free:free', note: 'free', provider: 'nous' }]
  }

  it('offers the recommended model for this session and names the credits way on', () => {
    const dialog = usageLimitDialog(limit as never)

    expect(dialog.title).toBe('Limit reached')
    expect(dialog.confirmLabel).toBe('Switch to Model Free (free)')
    expect(dialog.switchValue).toBe('example/model-free:free --provider nous --session')
    expect(dialog.detail).toBe("You've hit your daily limit · resets 15:56\nOr add credits with /topup.")
  })

  it('names the Keep going setting when a balance is waiting', () => {
    const dialog = usageLimitDialog({ ...limit, credits: { action: 'use_credits' } } as never)

    expect(dialog.detail.endsWith('Or turn on Keep going in /topup to use your credits.')).toBe(true)
  })

  it('names both steps when Keep going is off and there is no balance', () => {
    const dialog = usageLimitDialog({ ...limit, credits: { action: 'add_credits_and_keep_going' } } as never)

    expect(dialog.detail.endsWith('Or add credits and turn on Keep going in /topup.')).toBe(true)
  })

  it('without a model to offer, the confirm opens /topup', () => {
    const dialog = usageLimitDialog({ ...limit, suggestions: [] } as never)

    expect(dialog.confirmLabel).toBe('Open /topup')
    expect(dialog.switchValue).toBeNull()
  })
})

describe('translated picker words (TUI passes its own copy)', () => {
  const dayFull = window([{ label: 'day', left_fraction: 0, resets_at_ms: NOW + 3 * HOUR, scope: 'all' }])

  const de = {
    ...PLAN_LIMIT_WORDS_EN,
    resetsIn: (d: string) => `Zurücksetzung in ${d}`,
    limitReached: (name: string) => `${name} erreicht`,
    limitNames: {
      day: 'Tageslimit',
      month: 'Monatslimit',
      top_tier: 'Monatslimit für große Modelle',
      week: 'Wochenlimit'
    }
  }

  it('English stays the default', () => {
    expect(planLimitLine(dayFull, NOW)).toBe('Daily limit reached · resets in 3h')
  })

  it('a locale swaps every word', () => {
    expect(planLimitLine(dayFull, NOW, de)).toBe('Tageslimit erreicht · Zurücksetzung in 3h')
  })
})
