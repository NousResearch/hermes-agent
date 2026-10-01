/**
 * Billing/Subscription TUI fixture harness — renders any single overlay STATE
 * live in the terminal so it can be screenshotted (tmux) and UX-reviewed.
 *
 * This is a DEV/REVIEW tool, not shipped behaviour. It bypasses the gateway and
 * mounts the real Ink overlay components directly with a hand-built state object,
 * exactly the way the vitest render tests do — so what you see is pixel-identical
 * to what `/subscription` and `/topup` draw at runtime.
 *
 * Usage:
 *   npx tsx scripts/billing-fixtures.tsx <fixture-name>
 *   npx tsx scripts/billing-fixtures.tsx --list
 *
 * Drive a specific screen of a fixture with SCREEN=<screen>, e.g.:
 *   SCREEN=confirm npx tsx scripts/billing-fixtures.tsx sub-free
 *   SCREEN=handoff npx tsx scripts/billing-fixtures.tsx sub-mid
 *
 * The selection cursor can be moved with ↑/↓ once it's live (the components own
 * their own useInput); Esc/Enter behave as in production. Ctrl-C to exit.
 */
import { render } from '@hermes/ink'
import React from 'react'

import type { BillingOverlayState, SubscriptionOverlayState, SubscriptionScreen } from '../src/app/interfaces.js'
import { BillingOverlay } from '../src/components/billingOverlay.js'
import { SubscriptionOverlay } from '../src/components/subscriptionOverlay.js'
import type { BillingStateResponse, SubscriptionStateResponse, SubscriptionTierOption } from '../src/gatewayTypes.js'
import { DEFAULT_THEME } from '../src/theme.js'

const t = DEFAULT_THEME

// ── helpers ──────────────────────────────────────────────────────────

const tier = (o: Partial<SubscriptionTierOption> = {}): SubscriptionTierOption => ({
  tier_id: 'free',
  name: 'Free',
  tier_order: 0,
  dollars_per_month_display: '$0',
  monthly_credits: '0',
  is_current: false,
  is_enabled: true,
  ...o
})

// Mirrors the live portal catalog so fixtures don't drift; the real overlay
// reads tiers from GET /api/billing/subscription, never from here.
const TIERS = {
  free: tier({ tier_id: 'free', name: 'Free', tier_order: 0, dollars_per_month_display: '$0', monthly_credits: '0' }),
  plus: tier({ tier_id: 'plus', name: 'Plus', tier_order: 1, dollars_per_month_display: '$20', monthly_credits: '22' }),
  super: tier({ tier_id: 'super', name: 'Super', tier_order: 2, dollars_per_month_display: '$100', monthly_credits: '110' }),
  ultra: tier({ tier_id: 'ultra', name: 'Ultra', tier_order: 3, dollars_per_month_display: '$200', monthly_credits: '220' })
}

const tierList = (currentId?: string): SubscriptionTierOption[] =>
  Object.values(TIERS).map(x => ({ ...x, is_current: x.tier_id === currentId }))

const subState = (o: Partial<SubscriptionStateResponse> = {}): SubscriptionStateResponse => ({
  ok: true,
  logged_in: true,
  is_admin: true,
  can_change_plan: true,
  org_name: 'Acme Inc',
  org_id: 'org_acme',
  role: 'OWNER',
  context: 'personal',
  current: null,
  tiers: tierList(),
  portal_url: 'https://portal.nousresearch.com/billing',
  ...o
})

const cur = (o: Record<string, unknown> = {}) => ({
  tier_id: 'plus',
  tier_name: 'Plus',
  monthly_credits: '1000',
  credits_remaining: '420',
  cycle_ends_at: '2026-07-01',
  pending_downgrade_tier_name: null,
  pending_downgrade_at: null,
  cancel_at_period_end: false,
  cancellation_effective_at: null,
  ...o
})

const subCtx: SubscriptionOverlayState['ctx'] = {
  openManageLink: () => Promise.resolve(true),
  refreshState: () => Promise.resolve(null),
  sys: () => {}
}

const sub = (s: SubscriptionStateResponse, screen: SubscriptionScreen = 'overview', pendingTargetTierId: string | null = null): SubscriptionOverlayState => ({
  ctx: subCtx,
  screen,
  state: s,
  pendingTargetTierId
})

// ── billing/topup fixtures ───────────────────────────────────────────

const billState = (o: Partial<BillingStateResponse> = {}): BillingStateResponse => ({
  ok: true,
  logged_in: true,
  is_admin: true,
  cli_billing_enabled: true,
  can_charge: true,
  card: { brand: 'Visa', last4: '4242', masked: 'Visa •••• 4242' },
  balance_display: '$12.00',
  balance_usd: '12.00',
  min_usd: '5',
  max_usd: '500',
  monthly_cap: {
    is_default_ceiling: false,
    limit_display: '$20',
    limit_usd: '20',
    spent_display: '$8.00',
    spent_this_month_usd: '8'
  },
  auto_reload: { enabled: false, reload_to_display: '$25', reload_to_usd: '25', threshold_display: '$5', threshold_usd: '5' },
  org_name: 'Acme Inc',
  role: 'OWNER',
  portal_url: 'https://portal.nousresearch.com/billing',
  charge_presets: ['10', '25', '50', '100'],
  charge_presets_display: ['$10', '$25', '$50', '$100'],
  ...o
})

const billCtx = {
  applyAutoReload: () => Promise.resolve(true),
  charge: () => Promise.resolve('submitted' as const),
  openPortal: () => {},
  requestRemoteSpending: () => Promise.resolve(true),
  sys: () => {},
  validate: (raw: string) => ({ amount: raw })
}

const bill = (s: BillingStateResponse, screen: BillingOverlayState['screen'] = 'overview'): BillingOverlayState => ({
  ctx: billCtx,
  pendingCharge: screen === 'confirm' || screen === 'stepup' ? { amount: '100' } : null,
  screen,
  state: s
})

// ── fixture registry ─────────────────────────────────────────────────

type Fixture = { desc: string; node: React.ReactElement }

const subEl = (s: SubscriptionStateResponse, screen: SubscriptionScreen = 'overview', pending: string | null = null) =>
  React.createElement(SubscriptionOverlay, { onClose: () => {}, onPatch: () => {}, overlay: sub(s, screen, pending), t })

const billEl = (s: BillingStateResponse, screen: BillingOverlayState['screen'] = 'overview') =>
  React.createElement(BillingOverlay, { onClose: () => {}, onPatch: () => {}, overlay: bill(s, screen), t })

const FIXTURES: Record<string, Fixture> = {
  // /subscription — overview states
  'sub-free': {
    desc: 'Free / no sub — upgradeable (primary conversion state)',
    node: subEl(subState({ current: null }))
  },
  'sub-mid': {
    desc: 'Subscriber mid-tier (Plus) — usage bar + up/downgrade targets',
    node: subEl(subState({ current: cur(), tiers: tierList('plus') }))
  },
  'sub-top': {
    desc: 'Subscriber top-tier (Ultra) — "on the top plan"',
    node: subEl(subState({ current: cur({ tier_id: 'ultra', tier_name: 'Ultra', monthly_credits: '7000', credits_remaining: '5000' }), tiers: tierList('ultra') }))
  },
  'sub-not-admin': {
    desc: 'Member (not admin/owner) — read-only, no tier picker',
    node: subEl(subState({ is_admin: false, can_change_plan: false, role: 'MEMBER', current: cur(), tiers: tierList('plus') }))
  },
  'sub-downgrade': {
    desc: 'Downgrade scheduled — pending-switch banner',
    node: subEl(subState({ current: cur({ pending_downgrade_tier_name: 'Plus', pending_downgrade_at: '2026-07-15' }), tiers: tierList('super') }))
  },
  'sub-cancel': {
    desc: 'Cancellation scheduled — stays active until effective date',
    node: subEl(subState({ current: cur({ cancel_at_period_end: true, cancellation_effective_at: '2026-07-01' }), tiers: tierList('plus') }))
  },
  'sub-team': {
    desc: 'Team org context — shared credits, redirect to /topup',
    node: subEl(subState({ context: 'team', current: null, org_name: 'Acme Engineering' }))
  },
  // /subscription — non-overview screens
  'sub-confirm': {
    desc: 'Confirm plan change (deep-link, no in-terminal charge)',
    node: subEl(subState({ current: cur(), tiers: tierList('plus') }), 'confirm', 'super')
  },
  'sub-confirm-new': {
    desc: 'Confirm first subscription (free → paid)',
    node: subEl(subState({ current: null }), 'confirm', 'plus')
  },
  'sub-handoff': {
    desc: 'Handoff transient — opening subscription page in browser',
    node: subEl(subState({ current: cur() }), 'handoff')
  },
  // /topup (renamed /billing)
  'topup-overview': {
    desc: '/topup overview — admin, card on file, full menu',
    node: billEl(billState())
  },
  'topup-no-card': {
    desc: '/topup overview — admin, NO saved card (card hint)',
    node: billEl(billState({ card: null }))
  },
  'topup-not-admin': {
    desc: '/topup overview — member, read-only',
    node: billEl(billState({ is_admin: false }))
  },
  'topup-disabled': {
    desc: '/topup overview — remote spending OFF for org',
    node: billEl(billState({ cli_billing_enabled: false }))
  },
  'topup-buy': {
    desc: '/topup buy screen — presets',
    node: billEl(billState(), 'buy')
  },
  'topup-stepup': {
    desc: '/topup step-up — "Allow Remote Spending" (resumable, holds $100 buy)',
    node: billEl(billState(), 'stepup')
  }
}

// ── driver ───────────────────────────────────────────────────────────

const arg = process.argv[2]

if (!arg || arg === '--list' || arg === '-l') {
  const names = Object.keys(FIXTURES)
  process.stdout.write('Billing/Subscription TUI fixtures:\n\n')
  for (const name of names) {
    process.stdout.write(`  ${name.padEnd(18)} ${FIXTURES[name]!.desc}\n`)
  }
  process.stdout.write(`\n  ${names.length} fixtures. Run:  npx tsx scripts/billing-fixtures.tsx <name>\n`)
  process.exit(0)
}

const fixture = FIXTURES[arg]

if (!fixture) {
  process.stderr.write(`Unknown fixture: ${arg}\nRun with --list to see all.\n`)
  process.exit(1)
}

render(fixture.node)
