import type { BillingBlock } from '@hermes/shared/billing'

import { translate, type Locale } from '../i18n/index.js'

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
export function billingDialogCopy(block: BillingBlock, locale: Locale = 'en'): BillingDialogCopy {
  if (block.is_nous) {
    return {
      cancelLabel: translate(locale, 'billing.blocked.dismiss'),
      confirmLabel: translate(locale, 'billing.blocked.nous.confirm'),
      detail: translate(locale, 'billing.blocked.nous.detail'),
      title: translate(locale, 'billing.blocked.nous.title')
    }
  }

  const label = block.provider_label || translate(locale, 'billing.blocked.provider.fallbackLabel')

  return {
    cancelLabel: translate(locale, 'billing.blocked.dismiss'),
    confirmLabel: translate(
      locale,
      block.billing_url ? 'billing.blocked.provider.confirmOpen' : 'billing.blocked.provider.confirmSwitch'
    ),
    detail: translate(locale, 'billing.blocked.provider.detail', { provider: label }),
    title: translate(locale, 'billing.blocked.provider.title', { provider: label })
  }
}
