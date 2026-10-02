import type { BillingBlock } from '@hermes/shared/billing'

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
