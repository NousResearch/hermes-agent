// Billing overlay (`/topup`, credit purchases, card step-up). Owned namespace: `billing`.
// Leaves are strings or `(...args) => string`; packs supply `{0}`, `{1}` positional
// placeholders, so keep argument order stable and comment it when >1 arg.
//
// Amounts passed in are already formatted (`$25`, `$12.00`) — currency formatting
// is a value, not a translatable literal. Hotkey chords (Enter/Esc/↑/↓) and
// slash commands (/topup) stay as-is inside the surrounding keyed sentence.

export const billingEn = {
  billing: {
    common: {
      cancel: 'Cancel',
      back: 'Back',
      manageOnPortal: 'Manage on portal',
      noSavedCard: 'No saved card on file',
      payment: (card: string) => `Payment: ${card}`,
      invalid: 'invalid'
    },
    overview: {
      title: (balance: string) => `Top up · balance ${balance}`,
      org: (name: string) => `Org: ${name}`,
      // args: 0 = threshold, 1 = reload-to amount
      autoReloadOn: (threshold: string, reloadTo: string) => `Auto-reload: on (below ${threshold} → ${reloadTo})`,
      autoReloadOff: 'Auto-reload: off',
      needsBillingPermissions:
        'Billing actions need someone with billing permissions (owner, admin, or finance admin).',
      remoteSpendingOff:
        "Remote spending is off for this org — a billing admin can turn it on from the portal's Hermes Agent page.",
      addFunds: 'Add funds',
      autoReload: 'Auto-reload',
      monthlyLimit: 'Monthly limit',
      card: (card: string) => `Card: ${card}`,
      noCardHint: 'No saved card on file — “Add funds” walks you through adding one.',
      hint: (count: number) => `↑/↓ select · 1-${count} quick pick · Enter confirm · Esc close`
    },
    buy: {
      title: 'Add funds',
      addCardOnPortal: 'Add a card on the portal',
      checkAgain: 'I’ve added it — check again',
      customAmount: 'Custom amount…',
      refreshFailed: 'Could not refresh billing state — try again in a moment.',
      cardFound: (card: string) => `✓ Card found: ${card} — pick an amount.`,
      stillNoCard: 'Still no card on file — finish adding it on the portal, then check again.',
      invalidPreset: 'Invalid preset.',
      invalidAmount: 'Invalid amount.',
      addCardThenCheckAgain: 'Add a card on the billing page, then come back and pick “check again”.',
      portalLinkFailed: 'Could not build the portal link — is your portal configured?',
      enterCustomAmount: 'Enter a custom amount:',
      typingHint: 'Enter confirm · Esc back',
      noSavedCardLine: 'No saved card on file.',
      addCardOnce: 'Add a card once on the portal billing page — after that you can top up right from the terminal.',
      checkingForCard: 'Checking for a card…',
      hint: (count: number) => `↑/↓ select · 1-${count} quick pick · Enter confirm · Esc back`
    },
    confirm: {
      title: 'Confirm purchase',
      total: (amount: string) => `Total: ${amount}`,
      portalCardCharged: 'Your card saved on the portal will be charged.',
      authorization: 'By confirming, you allow Nous Research to charge your card.',
      payNow: (amount: string) => `Pay ${amount} now`,
      hint: '↑/↓ select · Enter confirm · Y/N quick · Esc back'
    },
    stepUp: {
      openingBrowser: 'Opening your browser to allow Remote Spending…',
      notAllowed:
        "! Couldn't allow Remote Spending — someone with billing permissions (owner, admin, or finance admin) has to approve it. Your card was not charged.",
      allowedResuming: '✓ Remote Spending allowed — resuming your purchase.',
      stillNeedsApproval:
        '! Remote Spending still needs approval — run /topup to try again. Your card was not charged.',
      declined: 'No charge made. Run /topup when you want to allow Remote Spending.',
      title: 'Allow Remote Spending',
      waitingForBrowser: 'Waiting for your browser…',
      approveInPage: 'Approve in the page that just opened.',
      heldHere: (amount: string) => `Your ${amount} top-up is held here and resumes when you're done.`,
      escCancel: 'Esc cancel',
      grantedTitle: 'Remote Spending allowed',
      readyToFinish: (amount: string) => `Your ${amount} top-up is ready to finish.`,
      pressEnterToResume: 'Press Enter to resume',
      grantedHint: 'Enter resume · Esc cancel',
      resuming: (amount: string) => `Resuming your ${amount} top-up…`,
      promptTitle: 'One-time setup',
      promptBody: 'To charge from this terminal, allow Remote Spending once.',
      promptDetail: (amount: string) =>
        `It opens your browser to authorize, then your ${amount} top-up picks up right here.`,
      allow: 'Allow Remote Spending',
      notNow: 'Not now',
      promptHint: '↑/↓ select · Enter confirm · Y/N quick · Esc cancel'
    },
    autoReload: {
      title: 'Auto-reload',
      description: 'Automatically add funds when your balance is low.',
      aDifferentCard: 'a different card',
      yourCard: 'your card',
      manageCardOnPortal: 'Use your card on file — manage on portal',
      agreeAndTurnOn: 'Agree and turn on',
      turnOff: 'Turn off',
      thresholdError: (error: string) => `Threshold: ${error}`,
      reloadToError: (error: string) => `Reload-to: ${error}`,
      reloadToMustExceedThreshold: 'Reload-to amount must be greater than the threshold.',
      noCardManageOnPortal: '🔴 No saved card — manage billing on the portal.',
      // args: 0 = threshold, 1 = reload-to amount
      turnedOn: (threshold: string, reloadTo: string) =>
        `✅ Auto-reload on: below ${threshold} → reload to ${reloadTo}.`,
      turnedOff: '✅ Auto-reload turned off.',
      cardOnFile: (card: string) => `Card on file: ${card}`,
      distinctCardWarning: (card: string) => `⚠ Auto-refill is charging ${card} — not your card on file.`,
      thresholdLabel: 'When balance falls below:',
      reloadToLabel: 'Reload balance to:',
      authorization: (card: string) =>
        `By confirming, you authorize Nous Research to charge ${card} whenever your balance falls below the threshold. Turn off any time here or on the portal.`,
      hint: '↑/↓ move · Tab switch field · Enter next/confirm · Esc back'
    },
    limit: {
      title: 'Monthly spend limit',
      // args: 0 = spent, 1 = limit
      usage: (spent: string, limit: string) => `${spent} of ${limit} used this month`,
      usageDefaultCeiling: (spent: string, limit: string) => `${spent} of ${limit} used this month (default ceiling)`,
      noCapVisible: 'No monthly cap visible (managed on the portal).',
      readOnly: 'The monthly limit is set on the portal — shown here read-only.',
      hint: (count: number) => `↑/↓ select · 1-${count} quick pick · Enter confirm · Esc back`
    }
  }
}
