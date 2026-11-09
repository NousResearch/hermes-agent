// Subscription overlay (`/subscription`: plans, scheduled changes, remote spending).
// Owned namespace: `subscription`. Leaves are strings or `(...args) => string`; packs
// supply `{0}`, `{1}` positional placeholders, so keep argument order stable and
// comment it when >1 arg.

export const subscriptionEn = {
  subscription: {
    // Fallbacks substituted for missing backend values + row labels shared by
    // several screens.
    shared: {
      periodEndFallback: 'the end of the billing period',
      thisPlan: 'this plan',
      yourNewPlan: 'your new plan',
      yourPlan: 'your plan',
      theSelectedPlan: 'the selected plan',
      freePlan: 'Free',
      back: 'Back',
      close: 'Close',
      cancel: 'Cancel',
      manageOnPortal: 'Manage on portal'
    },
    // Outcome messages produced by the RPC → result mappers.
    result: {
      genericError: 'Something went wrong. Try again, or manage on the portal.',
      upgradeUnconfirmed:
        'Couldn’t confirm the upgrade — your card may or may not have been charged. Re-run /subscription to check your plan before trying again.',
      verifyCard: 'Please verify your card in the portal to finish this upgrade.',
      cardDeclinedTryDifferent: 'Your card was declined — try a different card on the portal.',
      alreadyOn: (plan: string) => `You are already on ${plan}.`,
      upgraded: (plan: string) => `Upgraded to ${plan}. Your new monthly credits land in a moment.`,
      requiresAction: 'This upgrade needs extra verification (3DS). Finish it on the portal.',
      paymentFailed: 'Your card was declined. Update your payment method on the portal and try again.',
      scopeStillDenied:
        'Remote Spending still isn’t active for this terminal — the authorization didn’t take. Retry, or make this change on the portal.',
      couldNotPreview: 'Could not preview that change.',
      cancellationScheduled:
        'Scheduled — your plan stays active until the end of the billing period, then it cancels. Nothing changes today.',
      changeScheduled:
        'Scheduled — your plan doesn’t change today. You keep your current plan until the end of the billing period, then it switches.',
      resumed: 'Your pending change was undone — you stay on your current plan.'
    },
    // Typed denials from the remote-spending step-up.
    stepUpDenial: {
      sessionRevoked: 'Your session expired — run /portal to log in again, then retry the change.',
      remoteSpendingRevoked: 'Remote spending was stopped for this terminal — reconnect from the portal, then retry.',
      rateLimited: 'Too many attempts — wait a moment, then try again.',
      notAllowed:
        'Remote Spending was not allowed — someone with billing permissions (owner, admin, or finance admin) must approve it. You can also make this change on the portal.'
    },
    // The one-line plan status at the top of the overview.
    status: {
      freePlan: 'Plan: Free · free models only',
      plan: (plan: string) => `Plan: ${plan}`,
      transition: (to: string) => ` → ${to}`,
      left: (amount: string) => ` · ${amount} left`,
      renews: (date: string) => ` · renews ${date}`,
      viewOnly: ' · view only',
      // The pending-transition target when the subscription cancels at period end.
      cancels: 'cancels'
    },
    overview: {
      scheduledChange: '⏳ Scheduled change',
      keepUntilThen: (plan: string) => `You keep ${plan} (and its credits) until then.`,
      freeNudge: 'Paid models need a subscription. Start one to reach them.',
      lowNudge: (amount: string) => `Low balance · ${amount} left. Top up or upgrade before a mid-run cutoff.`,
      lowNudgeAmountFallback: 'under $5',
      noPortalUrl: '🔴 No portal URL available — manage your subscription on the Nous portal.',
      keepPlanUndo: (plan: string) => `Keep ${plan} (undo this change)`,
      changePlan: 'Change plan',
      cancelSubscription: 'Cancel subscription',
      // {0} tier name, {1} monthly price display (e.g. "$20")
      tierRow: (name: string, price: string) => `${name} · ${price}/mo`,
      creditsPerMonth: (credits: string) => ` · $${credits} credits/mo`,
      startSubscription: 'Start a subscription',
      org: (name: string) => `Org: ${name}`,
      hint: '↑/↓ select · Enter confirm · Esc close'
    },
    picker: {
      title: 'Change plan',
      current: (plan: string) => `Current: ${plan}. Pick a plan to see the effect before confirming.`,
      noOtherPlans: 'No other plans are available to switch to right now.',
      // {0} tier name, {1} monthly price display, {2} direction (upgrade/downgrade)
      tierRow: (name: string, price: string, direction: string) => `${name} · ${price}/mo · ${direction}`,
      upgrade: 'upgrade',
      downgrade: 'downgrade',
      hint: '↑/↓ select · Enter preview · Esc back'
    },
    confirm: {
      titleCancellation: 'Confirm cancellation',
      titleChange: 'Confirm plan change',
      chipChargedNow: 'charged now',
      chipScheduled: 'scheduled · not today',
      working: 'Working…',
      payAndUpgrade: (amount: string) => `Pay ${amount} & upgrade now`,
      upgradeNowProrated: 'Upgrade now (prorated charge)',
      scheduleChangeTo: (plan: string) => `Schedule change to ${plan}`,
      // {0} current plan name, {1} period-end date
      cancelBody: (plan: string, date: string) =>
        `Cancel ${plan} — it stays active until ${date}, then will not renew.`,
      cancelKeepCredits: 'You keep your remaining credits for this period. You can resume before it ends.',
      upgradeTo: (plan: string) => `Upgrade to ${plan}.`,
      chargedNow: (amount: string) => `You will be charged ${amount} now (prorated).`,
      chargedProrated: 'You will be charged the prorated amount now.',
      creditsDelta: (delta: string) => `Monthly credits change: ${delta}.`,
      cardCharged: (card: string) => `${card} — the card on your subscription — will be charged.`,
      cardChargedGeneric: 'The card on your subscription will be charged.',
      // {0} target plan name, {1} effective date
      scheduledBody: (plan: string, date: string) =>
        `Change to ${plan} — takes effect ${date}. No charge now; you keep your current plan until then.`,
      noOp: (plan: string) => `You are already on ${plan} — nothing to change.`,
      blockedFallback: 'That change cannot be made here — manage it on the portal.',
      hint: '↑/↓ select · Enter confirm · Esc back'
    },
    resultScreen: {
      titleApplying: 'Applying…',
      titleStillApplying: 'Still applying',
      titleDone: 'Done',
      titleFailed: 'Could not complete',
      stillApplying: 'Your upgrade succeeded and is still applying — refresh in a moment.',
      rerunHint: 'Re-run /subscription anytime to review it.',
      openPortalToFinish: 'Open the portal to finish',
      hint: '↑/↓ select · Enter · Esc close'
    },
    stepUp: {
      title: 'Remote Spending',
      promptBody: 'Changing your plan needs Remote Spending allowed for this terminal. Allow it here, then continue.',
      promptNote: 'Someone with billing permissions (owner, admin, or finance admin) approves it once in the browser.',
      waiting: 'Opening your browser to approve… finish there, then come back — nothing is charged until you continue.',
      granted: 'Remote Spending allowed. Continue to finish your change.',
      resuming: 'Applying your change…',
      allow: 'Allow Remote Spending',
      continue: 'Continue',
      continueChange: 'Continue the change',
      hintWaiting: 'Waiting for approval… · Esc to cancel',
      hintWorking: 'Working…',
      hint: '↑/↓ select · Enter · Esc back'
    },
    team: {
      title: 'Team subscription',
      orgFallback: 'a team org',
      body: (org: string) =>
        `This terminal is connected to ${org}. Teams run on a shared balance · use /topup to add funds.`,
      personalNote: 'Personal subscriptions live on your personal account.',
      hint: 'Enter/Esc close'
    }
  }
}
