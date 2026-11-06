import type { BillingBlock } from '@hermes/shared/billing'
import type { UsageLimit } from '@hermes/shared/plan-usage'

import { t } from '../i18n/runtime.js'

export interface BillingDialogCopy {
  cancelLabel: string
  confirmLabel: string
  detail: string
  title: string
}

/**
 * Copy for the out-of-credits confirm dialog (the TUI's billing wall). The
 * dialog is the actionable layer — the full provider guidance already lands in
 * the transcript — so `detail` stays to one concise, non-truncating line and the
 * confirm button carries the recovery: Nous → `/topup`, other providers → their
 * billing page (or `/model` to switch when we have no URL). Pure + exported so
 * the wording is unit-tested without driving the gateway.
 */
export function billingDialogCopy(block: BillingBlock): BillingDialogCopy {
  if (block.is_nous) {
    return {
      cancelLabel: t('libText.billingDialog.dismiss'),
      confirmLabel: t('libText.billingDialog.topUp'),
      detail: t('libText.billingDialog.nousDetail'),
      title: t('libText.billingDialog.nousTitle')
    }
  }

  const label = block.provider_label || t('libText.billingDialog.yourProvider')

  return {
    cancelLabel: t('libText.billingDialog.dismiss'),
    confirmLabel: block.billing_url
      ? t('libText.billingDialog.openBillingPage')
      : t('libText.billingDialog.switchProvider'),
    detail: t('libText.billingDialog.providerDetail', label),
    title: t('libText.billingDialog.providerTitle', label)
  }
}

export interface UsageLimitDialog extends BillingDialogCopy {
  /** `/model` argument for the switch (this session only), or null: open /topup instead. */
  switchValue: null | string
}

// agent/plan_limits.py::credits_action → the dialog's credits line.
const CREDITS_STEP: Record<string, () => string> = {
  add_credits: () => t('libText.billingDialog.limitCreditsAdd'),
  add_credits_and_keep_going: () => t('libText.billingDialog.limitCreditsBoth'),
  use_credits: () => t('libText.billingDialog.limitCreditsUse')
}

/**
 * A usage-window plan's limit (error_surface.usage_limit, agent/plan_limits.py): the confirm button
 * switches THIS session to the plan's recommended model and re-sends the message; the detail names the
 * other way on (credits, via /topup). With no model to offer, the confirm opens /topup instead. Never
 * switches on its own: the dialog is the ask.
 */
export function usageLimitDialog(limit: UsageLimit): UsageLimitDialog {
  const suggestion = limit.suggestions?.[0]
  const sentence = (limit.message || '').trim()

  const action = limit.credits?.action
  const credits = action && Object.hasOwn(CREDITS_STEP, action) ? CREDITS_STEP[action]!() : ''

  if (!suggestion) {
    return {
      cancelLabel: t('libText.billingDialog.dismiss'),
      confirmLabel: t('libText.billingDialog.limitOpenTopup'),
      switchValue: null,
      detail: [sentence, credits].filter(Boolean).join('\n'),
      title: t('libText.billingDialog.limitTitle')
    }
  }

  const note = suggestion.note ? ` (${suggestion.note})` : ''

  return {
    cancelLabel: t('libText.billingDialog.dismiss'),
    confirmLabel: t('libText.billingDialog.limitSwitchTo', `${suggestion.label}${note}`),
    detail: [sentence, credits].filter(Boolean).join('\n'),
    switchValue: `${suggestion.model} --provider ${suggestion.provider || 'nous'} --session`,
    title: t('libText.billingDialog.limitTitle')
  }
}
