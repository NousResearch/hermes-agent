import { Box, Text, useInput } from '@hermes/ink'
import { useState } from 'react'

import type { BillingOverlayState } from '../app/interfaces.js'
import type { BillingStateResponse } from '../gatewayTypes.js'
import { type I18nApi, useI18n } from '../i18n/index.js'
import type { Theme } from '../theme.js'

import { TextInput } from './textInput.js'

const SPEND_BAR_CELLS = 10

type Translator = I18nApi['t']

interface BillingOverlayProps {
  /** Replace the overlay slot (screen transitions + pending data). */
  onPatch: (next: Partial<BillingOverlayState>) => void
  /** Close the overlay entirely. */
  onClose: () => void
  overlay: BillingOverlayState
  t: Theme
}

/** A numbered menu row with the ▸ cursor (mirrors ClarifyPrompt). */
function MenuRow({ active, index, label, t }: { active: boolean; index: number; label: string; t: Theme }) {
  return (
    <Text>
      <Text bold={active} color={active ? t.color.label : t.color.muted} inverse={active}>
        {active ? '▸ ' : '  '}
        {index}. {label}
      </Text>
    </Text>
  )
}

/** Plain (non-numbered) action row with the ▸ cursor (confirm screens). */
function ActionRow({ active, label, color, t }: { active: boolean; label: string; color?: string; t: Theme }) {
  return (
    <Text>
      <Text color={active ? t.color.accent : t.color.muted}>{active ? '▸ ' : '  '}</Text>
      <Text bold={active} color={active ? (color ?? t.color.text) : t.color.muted}>
        {label}
      </Text>
    </Text>
  )
}

/** 10-cell spend bar + percent (omit entirely when there's no usable cap). */
function spendBar(s: BillingStateResponse, tr: Translator): null | string {
  const cap = s.monthly_cap

  if (!cap || cap.limit_usd == null) {
    return null
  }

  const limit = Number(cap.limit_usd)
  const spent = Number(cap.spent_this_month_usd ?? '0')

  if (!(limit > 0) || Number.isNaN(spent)) {
    return null
  }

  const ratio = Math.max(0, Math.min(1, spent / limit))
  const filled = Math.round(ratio * SPEND_BAR_CELLS)
  const bar = '█'.repeat(filled) + '░'.repeat(SPEND_BAR_CELLS - filled)
  const pct = Math.round(ratio * 100)
  const ceiling = cap.is_default_ceiling ? tr('billing.spend.defaultCeilingSuffix') : ''

  return tr('billing.spend.line', {
    bar,
    ceiling,
    limit: cap.limit_display,
    percent: pct,
    spent: cap.spent_display
  })
}

function autoReloadLine(s: BillingStateResponse, tr: Translator): null | string {
  if (!s.auto_reload) {
    return null
  }

  return s.auto_reload.enabled
    ? tr('billing.autoReload.lineOn', {
        reloadTo: s.auto_reload.reload_to_display,
        threshold: s.auto_reload.threshold_display
      })
    : tr('billing.autoReload.lineOff')
}

const footer = (extra: string, t: Theme) => <Text color={t.color.muted}>{extra}</Text>

/**
 * The /billing modal.  A self-contained state machine:
 *   overview → buy | autoreload | limit  (and buy → confirm).
 * Esc from a sub-screen returns to overview; Esc from overview closes.
 * All RPCs + error mapping live in billing.ts and are reached through
 * `overlay.ctx` — this component only renders + routes keys.
 */
export function BillingOverlay({ onClose, onPatch, overlay, t }: BillingOverlayProps) {
  const { ctx, screen, state: s } = overlay
  const i18n = useI18n()
  const tr = i18n.t

  return (
    <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" paddingX={1}>
      {screen === 'overview' && <OverviewScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} tr={tr} />}
      {screen === 'buy' && <BuyScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} tr={tr} />}
      {screen === 'confirm' && (
        <ConfirmScreen
          amount={overlay.pendingCharge?.amount ?? ''}
          ctx={ctx}
          onBack={() => onPatch({ pendingCharge: null, screen: 'buy' })}
          onClose={onClose}
          s={s}
          t={t}
          tr={tr}
        />
      )}
      {screen === 'autoreload' && (
        <AutoReloadScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} tr={tr} />
      )}
      {screen === 'limit' && <LimitScreen ctx={ctx} onClose={onClose} onPatch={onPatch} s={s} t={t} tr={tr} />}
    </Box>
  )
}

// ── Screen 1: Overview ────────────────────────────────────────────────

interface ScreenProps {
  ctx: BillingOverlayState['ctx']
  onClose: () => void
  onPatch: (next: Partial<BillingOverlayState>) => void
  s: BillingStateResponse
  t: Theme
  tr: Translator
}

type OverviewAction = 'autoreload' | 'buy' | 'cancel' | 'limit' | 'portal'

function OverviewScreen({ ctx, onClose, onPatch, s, t, tr }: ScreenProps) {
  // Gate: full menu only for an admin with the kill-switch on. Otherwise the
  // menu collapses to Manage-on-portal / Cancel + a one-line note.
  const full = s.is_admin && s.cli_billing_enabled

  const note = !s.is_admin
    ? tr('billing.overview.adminRequired')
    : !s.cli_billing_enabled
      ? tr('billing.overview.terminalOff')
      : null

  // Optimistic funnel: admin + kill-switch on but no saved card → a charge will
  // 403 no_payment_method. Advise up front (Buy stays available — /state.card
  // can't fully prove CLI-chargeability, so we hint rather than hide).
  const cardHint = full && !s.card ? tr('billing.overview.noCardHint') : null

  const items = full
    ? ([
        { id: 'buy', label: tr('billing.action.buyCredits') },
        { id: 'autoreload', label: tr('billing.action.adjustAutoReload') },
        { id: 'limit', label: tr('billing.action.adjustMonthlyLimit') },
        { id: 'portal', label: tr('billing.action.managePortal') },
        { id: 'cancel', label: tr('common.cancel') }
      ] satisfies Array<{ id: OverviewAction; label: string }>)
    : ([
        { id: 'portal', label: tr('billing.action.managePortal') },
        { id: 'cancel', label: tr('common.cancel') }
      ] satisfies Array<{ id: OverviewAction; label: string }>)

  const [sel, setSel] = useState(0)

  const choose = (i: number) => {
    const action = items[i]?.id

    if (action === 'buy') {
      onPatch({ screen: 'buy' })
    } else if (action === 'autoreload') {
      onPatch({ screen: 'autoreload' })
    } else if (action === 'limit') {
      onPatch({ screen: 'limit' })
    } else if (action === 'portal') {
      if (s.portal_url) {
        ctx.openPortal(s.portal_url)
      }

      onClose()
    } else {
      onClose()
    }
  }

  useInput((ch, key) => {
    if (key.escape) {
      return onClose()
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
    }

    if (key.downArrow && sel < items.length - 1) {
      setSel(v => v + 1)
    }

    if (key.return) {
      return choose(sel)
    }

    const n = parseInt(ch, 10)

    if (n >= 1 && n <= items.length) {
      return choose(n - 1)
    }
  })

  const bar = spendBar(s, tr)
  const auto = autoReloadLine(s, tr)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {tr('billing.title.usageCredits')}
      </Text>
      {bar && <Text color={t.color.text}>{bar}</Text>}
      <Text color={t.color.text}>{tr('billing.balanceLine', { balance: s.balance_display })}</Text>
      {auto && <Text color={t.color.muted}>{auto}</Text>}
      {s.org_name && (
        <Text color={t.color.muted}>
          {tr('billing.orgLine', { org: s.org_name })}
          {s.role ? ` · ${s.role}` : ''}
        </Text>
      )}
      {note && (
        <Box marginTop={1}>
          <Text color={t.color.warn}>{note}</Text>
        </Box>
      )}
      {cardHint && (
        <Box marginTop={1}>
          <Text color={t.color.warn}>{cardHint}</Text>
        </Box>
      )}
      {cardHint && s.portal_url && <Text color={t.color.muted}>{tr('billing.portalLine', { url: s.portal_url })}</Text>}

      <Text />
      {items.map((item, i) => (
        <MenuRow active={sel === i} index={i + 1} key={item.id} label={item.label} t={t} />
      ))}

      <Text />
      {footer(tr('billing.footer.overview', { count: items.length }), t)}
    </Box>
  )
}

// ── Screen 2: Buy credits ─────────────────────────────────────────────

type BuyRow = { id: 'cancel' | 'custom'; label: string } | { id: 'preset'; index: number; label: string }

const paymentLine = (s: BillingStateResponse, tr: Translator) =>
  s.card ? tr('billing.payment.card', { card: s.card.masked }) : tr('billing.payment.noSavedCard')

function BuyScreen({ ctx, onPatch, s, t, tr }: ScreenProps) {
  const presets = s.charge_presets_display
  const rawPresets = s.charge_presets

  // rows: [...presets, custom amount, cancel]
  const rows: BuyRow[] = [
    ...presets.map((label, index) => ({ id: 'preset' as const, index, label })),
    { id: 'custom', label: tr('billing.buy.customAmount') },
    { id: 'cancel', label: tr('common.cancel') }
  ]

  const customIdx = presets.length

  const [sel, setSel] = useState(0)
  const [typing, setTyping] = useState(false)
  const [custom, setCustom] = useState('')
  const [error, setError] = useState<null | string>(null)

  const toConfirm = (amount: string) => {
    onPatch({ pendingCharge: { amount }, screen: 'confirm' })
  }

  const pickPreset = (i: number) => {
    // Prefer the raw (numeric) preset for the amount; fall back to stripping $.
    const raw = (rawPresets[i] ?? presets[i] ?? '').replace(/^\$/, '').trim()
    const v = ctx.validate(raw)

    if (v.error || !v.amount) {
      setError(v.error ?? tr('billing.buy.invalidPreset'))

      return
    }

    toConfirm(v.amount)
  }

  const submitCustom = (raw: string) => {
    const v = ctx.validate(raw)

    if (v.error || !v.amount) {
      setError(v.error ?? tr('billing.buy.invalidAmount'))

      return
    }

    toConfirm(v.amount)
  }

  const choose = (i: number) => {
    if (i < presets.length) {
      pickPreset(i)
    } else if (i === customIdx) {
      setError(null)
      setTyping(true)
    } else {
      onPatch({ screen: 'overview' })
    }
  }

  useInput((ch, key) => {
    if (key.escape) {
      return typing ? (setTyping(false), setError(null)) : onPatch({ screen: 'overview' })
    }

    if (typing) {
      return
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
    }

    if (key.downArrow && sel < rows.length - 1) {
      setSel(v => v + 1)
    }

    if (key.return) {
      return choose(sel)
    }

    const n = parseInt(ch, 10)

    if (n >= 1 && n <= rows.length) {
      return choose(n - 1)
    }
  })

  const payLine = paymentLine(s, tr)

  if (typing) {
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.accent}>
          {tr('billing.title.buyCredits')}
        </Text>
        <Text color={t.color.muted}>{payLine}</Text>
        <Text />
        <Text color={t.color.label}>{tr('billing.buy.enterCustomAmount')}</Text>
        <Box>
          <Text color={t.color.label}>{'$'}</Text>
          <TextInput columns={20} onChange={setCustom} onSubmit={submitCustom} value={custom} />
        </Box>
        {error && <Text color={t.color.error}>{error}</Text>}
        <Text />
        {footer(tr('billing.footer.confirmBack'), t)}
      </Box>
    )
  }

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {tr('billing.title.buyCredits')}
      </Text>
      <Text color={t.color.muted}>{payLine}</Text>
      <Text />
      {rows.map((row, i) => (
        <MenuRow active={sel === i} index={i + 1} key={`${row.id}-${i}`} label={row.label} t={t} />
      ))}
      {error && <Text color={t.color.error}>{error}</Text>}
      <Text />
      {footer(tr('billing.footer.pickBack', { count: rows.length }), t)}
    </Box>
  )
}

// ── Screen 3: Confirm purchase ────────────────────────────────────────

function ConfirmScreen({
  amount,
  ctx,
  onBack,
  onClose,
  s,
  t,
  tr
}: {
  amount: string
  ctx: BillingOverlayState['ctx']
  onBack: () => void
  onClose: () => void
  s: BillingStateResponse
  t: Theme
  tr: Translator
}) {
  // rows: Pay $X now / Cancel
  const [sel, setSel] = useState(0)

  const pay = () => {
    ctx.charge(amount)
    // Settlement is reported via transcript lines; close the overlay now.
    onClose()
  }

  const back = () => onBack()

  useInput((ch, key) => {
    if (key.escape) {
      return back()
    }

    const lower = ch.toLowerCase()

    if (lower === 'y') {
      return pay()
    }

    if (lower === 'n') {
      return back()
    }

    if (key.upArrow) {
      setSel(0)
    }

    if (key.downArrow) {
      setSel(1)
    }

    if (key.return) {
      return sel === 0 ? pay() : back()
    }
  })

  const payLine = paymentLine(s, tr)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {tr('billing.title.confirmPurchase')}
      </Text>
      <Text color={t.color.text}>{tr('billing.confirm.total', { amount })}</Text>
      <Text color={t.color.muted}>{payLine}</Text>
      <Text color={t.color.muted}>{tr('billing.confirm.consent')}</Text>
      <Text />
      <ActionRow active={sel === 0} color={t.color.ok} label={tr('billing.confirm.payNow', { amount })} t={t} />
      <ActionRow active={sel === 1} label={tr('common.cancel')} t={t} />
      <Text />
      {footer(tr('billing.footer.confirmYesNo'), t)}
    </Box>
  )
}

// ── Screen 4: Auto-reload (the 2-field form) ──────────────────────────

type AutoReloadAction = 'cancel' | 'turnOff' | 'turnOn'

function AutoReloadScreen({ ctx, onClose, onPatch, s, t, tr }: ScreenProps) {
  const ar = s.auto_reload
  const enabled = Boolean(ar?.enabled)

  // Prefill from state (strip the $ from the *_usd raw fields if present).
  const prefill = (raw?: null | string) => (raw == null ? '' : String(raw).replace(/^\$/, '').trim())
  const [threshold, setThreshold] = useState(prefill(ar?.threshold_usd))
  const [reloadTo, setReloadTo] = useState(prefill(ar?.reload_to_usd))
  const [field, setField] = useState<'reloadTo' | 'threshold'>('threshold')
  const [error, setError] = useState<null | string>(null)

  // focusRow: 0=threshold field, 1=reloadTo field, 2=Agree, 3=Turn off (if enabled), last=Cancel
  const actionRows = enabled
    ? ([
        { id: 'turnOn', label: tr('billing.autoReload.turnOn') },
        { id: 'turnOff', label: tr('billing.autoReload.turnOff') },
        { id: 'cancel', label: tr('common.cancel') }
      ] satisfies Array<{ id: AutoReloadAction; label: string }>)
    : ([
        { id: 'turnOn', label: tr('billing.autoReload.turnOn') },
        { id: 'cancel', label: tr('common.cancel') }
      ] satisfies Array<{ id: AutoReloadAction; label: string }>)

  const FIELD_ROWS = 2
  const [row, setRow] = useState(0)

  const noCard = !s.card

  const validatePair = (): null | { reloadTo: string; threshold: string } => {
    const tv = ctx.validate(threshold)

    if (tv.error || !tv.amount) {
      setError(
        tr('billing.autoReload.fieldError', {
          field: tr('billing.autoReload.thresholdShort'),
          message: tv.error ?? tr('billing.invalid')
        })
      )

      return null
    }

    const rv = ctx.validate(reloadTo)

    if (rv.error || !rv.amount) {
      setError(
        tr('billing.autoReload.fieldError', {
          field: tr('billing.autoReload.reloadToShort'),
          message: rv.error ?? tr('billing.invalid')
        })
      )

      return null
    }

    if (Number(rv.amount) <= Number(tv.amount)) {
      setError(tr('billing.autoReload.reloadAboveThreshold'))

      return null
    }

    setError(null)

    return { reloadTo: rv.amount, threshold: tv.amount }
  }

  const turnOn = () => {
    if (noCard) {
      ctx.sys(tr('billing.autoReload.noCard'))

      if (s.portal_url) {
        ctx.openPortal(s.portal_url)
      }

      onClose()

      return
    }

    const pair = validatePair()

    if (!pair) {
      return
    }

    void ctx.applyAutoReload(true, Number(pair.threshold), Number(pair.reloadTo)).then(ok => {
      if (ok) {
        ctx.sys(tr('billing.autoReload.enabled', { reloadTo: pair.reloadTo, threshold: pair.threshold }))
      }
    })
    onClose()
  }

  const turnOff = () => {
    void ctx.applyAutoReload(false).then(ok => {
      if (ok) {
        ctx.sys(tr('billing.autoReload.disabled'))
      }
    })
    onClose()
  }

  const onAction = (action: AutoReloadAction) => {
    if (action === 'turnOn') {
      turnOn()
    } else if (action === 'turnOff') {
      turnOff()
    } else {
      onPatch({ screen: 'overview' })
    }
  }

  const editingField = row < FIELD_ROWS

  useInput((ch, key) => {
    if (key.escape) {
      return onPatch({ screen: 'overview' })
    }

    if (key.upArrow && row > 0) {
      setRow(v => v - 1)
      setField(row - 1 === 0 ? 'threshold' : 'reloadTo')
    }

    if (key.downArrow && row < FIELD_ROWS + actionRows.length - 1) {
      setRow(v => v + 1)
      setField(row + 1 === 0 ? 'threshold' : 'reloadTo')
    }

    // Tab cycles between the two fields when focused on a field.
    if (key.tab && editingField) {
      const next = field === 'threshold' ? 'reloadTo' : 'threshold'
      setField(next)
      setRow(next === 'threshold' ? 0 : 1)
    }

    if (key.return && !editingField) {
      const idx = row - FIELD_ROWS

      return onAction(actionRows[idx]?.id ?? 'cancel')
    }

    // a number quick-picks an action row (1..actionRows.length)
    if (!editingField) {
      const n = parseInt(ch, 10)

      if (n >= 1 && n <= actionRows.length) {
        return onAction(actionRows[n - 1]!.id)
      }
    }
  })

  const cardLine = s.card
    ? tr('billing.payment.cardOnFile', { card: s.card.masked })
    : tr('billing.payment.noSavedCard')

  const fieldBox = (label: string, value: string, onChange: (v: string) => void, focused: boolean, key: string) => (
    <Box flexDirection="column" key={key}>
      <Text color={focused ? t.color.label : t.color.muted}>{label}</Text>
      <Box borderColor={focused ? t.color.accent : t.color.border} borderStyle="round" paddingX={1}>
        <Text color={t.color.label}>{'$'}</Text>
        <TextInput
          columns={16}
          focus={focused}
          onChange={onChange}
          onSubmit={() => {
            // Enter inside the threshold field jumps to reload-to; inside
            // reload-to jumps to the Agree action.
            if (key === 'threshold') {
              setField('reloadTo')
              setRow(1)
            } else {
              setRow(FIELD_ROWS)
            }
          }}
          value={value}
        />
      </Box>
    </Box>
  )

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {tr('billing.title.autoReload')}
      </Text>
      <Text color={t.color.muted}>{tr('billing.autoReload.description')}</Text>
      <Text color={t.color.muted}>{cardLine}</Text>
      <Text />
      {fieldBox(tr('billing.autoReload.thresholdLabel'), threshold, setThreshold, row === 0, 'threshold')}
      {fieldBox(tr('billing.autoReload.reloadToLabel'), reloadTo, setReloadTo, row === 1, 'reloadTo')}
      <Text />
      <Text color={t.color.muted}>
        {tr('billing.autoReload.authorization', {
          card: s.card ? s.card.masked : tr('billing.payment.yourCard')
        })}
      </Text>
      {error && <Text color={t.color.error}>{error}</Text>}
      <Text />
      {actionRows.map((action, i) => (
        <ActionRow
          active={!editingField && row - FIELD_ROWS === i}
          color={action.id === 'turnOff' ? t.color.warn : action.id === 'turnOn' ? t.color.ok : t.color.text}
          key={action.id}
          label={action.label}
          t={t}
        />
      ))}
      <Text />
      {footer(tr('billing.footer.autoReload'), t)}
    </Box>
  )
}

// ── Screen 5: Monthly spend limit (read-only) ─────────────────────────

type LimitAction = 'cancel' | 'portal'

function LimitScreen({ ctx, onClose, onPatch, s, t, tr }: ScreenProps) {
  const rows = [
    { id: 'portal', label: tr('billing.action.managePortal') },
    { id: 'cancel', label: tr('common.cancel') }
  ] satisfies Array<{ id: LimitAction; label: string }>

  const [sel, setSel] = useState(0)

  const choose = (i: number) => {
    if (rows[i]?.id === 'portal' && s.portal_url) {
      ctx.openPortal(s.portal_url)

      return onClose()
    }

    onPatch({ screen: 'overview' })
  }

  useInput((ch, key) => {
    if (key.escape) {
      return onPatch({ screen: 'overview' })
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
    }

    if (key.downArrow && sel < rows.length - 1) {
      setSel(v => v + 1)
    }

    if (key.return) {
      return choose(sel)
    }

    const n = parseInt(ch, 10)

    if (n >= 1 && n <= rows.length) {
      return choose(n - 1)
    }
  })

  const cap = s.monthly_cap

  const usageLine =
    cap && cap.limit_usd != null
      ? tr('billing.limit.usageLine', {
          ceiling: cap.is_default_ceiling ? tr('billing.spend.defaultCeilingSuffix') : '',
          limit: cap.limit_display,
          spent: cap.spent_display
        })
      : tr('billing.limit.noCap')

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.accent}>
        {tr('billing.title.monthlyLimit')}
      </Text>
      <Text color={t.color.text}>{usageLine}</Text>
      <Text color={t.color.muted}>{tr('billing.limit.readOnly')}</Text>
      <Text />
      {rows.map((row, i) => (
        <MenuRow active={sel === i} index={i + 1} key={row.id} label={row.label} t={t} />
      ))}
      <Text />
      {footer(tr('billing.footer.pickBack', { count: rows.length }), t)}
    </Box>
  )
}
