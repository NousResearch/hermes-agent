import type { ModelOptionProvider } from '@hermes/shared/gateway-events'
import {
  blockingGauge,
  formatResetsIn,
  formatUsageMultiplier,
  PLAN_LIMIT_WORDS_EN,
  planBlocks,
  planLimitLine,
  usageBasis,
  usageChip,
  usageTone
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
  usage_basis: 'a/low',
  usage_basis_name: 'Low Model',
  usage_free_models: ['a/free:free', 'a/zero'],
  usage_multipliers: { 'a/large': 40, 'a/low': 1, 'a/tiny': 0.25 }
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

  it('a full top-tier gauge blocks no model', () => {
    expect(planBlocks(topFull, 'a/large')).toBe(false)
    expect(planBlocks(topFull, 'a/low')).toBe(false)
  })

  it('free models are never used up (free catalog row, :free id)', () => {
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

  it('non-window accounts are never blocked or chipped', () => {
    const credits = { plan_usage: { kind: 'credits' }, usage_basis: 'a/low', usage_multipliers: { 'a/low': 1 } }

    expect(planBlocks({ plan_usage: { kind: 'credits' } } as Provider, 'a/low')).toBe(false)
    expect(planBlocks(undefined, 'a/low')).toBe(false)
    expect(usageChip(credits as Provider, 'a/low')).toBeNull()
    expect(usageBasis(credits as Provider, ['a/low'])).toBeNull()
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

  it("row chips: the gateway's multiplier, Free wins, nothing derived", () => {
    expect(usageChip(dayFull, 'a/large')).toEqual({ kind: 'multiplier', value: 40 })
    expect(usageChip(dayFull, 'a/low')).toEqual({ kind: 'multiplier', value: 1 })
    expect(usageChip(dayFull, 'a/free:free')).toEqual({ kind: 'free' })
    expect(usageChip(dayFull, 'a/unpriced')).toBeNull()
    // Free wins over any multiplier; malformed values are ignored.
    expect(usageChip({ ...dayFull, usage_multipliers: { 'b/x:free': 3 } }, 'b/x:free')).toEqual({ kind: 'free' })

    for (const bad of [0, -1, 100_001, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(usageChip({ ...dayFull, usage_multipliers: { 'b/x': bad } }, 'b/x')).toBeNull()
    }

    expect(usageChip({ ...dayFull, usage_multipliers: { 'b/x': 100_000 } }, 'b/x')).toEqual({
      kind: 'multiplier',
      value: 100_000
    })
  })

  it('formats the multiplier as sent: <0.1× below 0.1, one decimal below 1, else N×', () => {
    expect(formatUsageMultiplier(0.05)).toBe('<0.1×')
    expect(formatUsageMultiplier(0.1)).toBe('0.1×')
    expect(formatUsageMultiplier(0.4)).toBe('0.4×')
    expect(formatUsageMultiplier(0.25)).toBe('0.3×')
    expect(formatUsageMultiplier(0.1 + 0.2)).toBe('0.3×')
    expect(formatUsageMultiplier(1)).toBe('1×')
    expect(formatUsageMultiplier(7)).toBe('7×')
    expect(formatUsageMultiplier(15)).toBe('15×')
  })

  it('tones a multiplier by fixed cutoffs: plain under 15×, amber from 15×, red from 50×', () => {
    expect([0.4, 1, 10, 14].map(usageTone)).toEqual(['low', 'low', 'low', 'low'])
    expect([15, 20, 40, 45].map(usageTone)).toEqual(['mid', 'mid', 'mid', 'mid'])
    expect([50, 55, 110, 1000].map(usageTone)).toEqual(['high', 'high', 'high', 'high'])
  })

  it('the basis line: the gateway name whenever a multiplier shows, listed or not', () => {
    const named = { id: 'a/low', name: 'Low Model' }
    expect(usageBasis(dayFull, ['a/low', 'a/large'])).toEqual(named)
    // The basis row need not be listed (an org allowlist can hide it).
    expect(usageBasis(dayFull, ['a/large'])).toEqual(named)
    expect(usageBasis(dayFull, ['a/low', 'a/free:free'])).toEqual(named)
    expect(usageBasis({ ...dayFull, usage_multipliers: {} }, ['a/low'])).toBeNull()
    expect(usageBasis({ ...dayFull, usage_basis: null }, ['a/large'])).toBeNull()
    // No name from the gateway: the id.
    expect(usageBasis({ ...dayFull, usage_basis_name: null }, ['a/large'])).toEqual({ id: 'a/low', name: 'a/low' })
  })

  it('a usage_levels payload shows only Free', () => {
    const old = {
      plan_usage: { gauges: [], kind: 'usage_window' },
      usage_free_models: ['a/zero'],
      usage_levels: { 'a/large': 'top_tier', 'a/zero': 'free' }
    } as Provider

    expect(usageChip(old, 'a/large')).toBeNull()
    expect(usageChip(old, 'a/zero')).toEqual({ kind: 'free' })
  })

  it('subscription-billed rows: never dimmed, no Free chip', () => {
    const subscribed: Provider = {
      ...dayFull,
      plan_usage: { ...dayFull.plan_usage!, subscription_model_ids: ['a/zero'] }
    }

    expect(planBlocks(subscribed, 'a/zero')).toBe(false)
    expect(usageChip(subscribed, 'a/zero')).toBeNull()
    expect(usageChip(subscribed, 'a/free:free')).toEqual({ kind: 'free' })
    expect(usageChip(subscribed, 'a/large')).toEqual({ kind: 'multiplier', value: 40 })
    // Without the field: Free.
    expect(usageChip(dayFull, 'a/zero')).toEqual({ kind: 'free' })
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
