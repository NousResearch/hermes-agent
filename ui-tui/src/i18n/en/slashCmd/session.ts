// slashCmd.session / slashCmd.topup / slashCmd.subscription — replies and usage hints of
// app/slash/commands/{session,topup,subscription}.ts. Command NAMES/aliases/arg syntax
// stay literal; config VALUES echoed back (auto/light/dark, fast/normal, hide/show, …)
// are identifiers and are interpolated, not translated.

export const slashCmdSessionEn = {
  session: {
    bg: {
      started: (taskId: string) => `bg ${taskId} started`,
      usage: '/bg <prompt>'
    },
    branch: {
      branched: (title: string) => `branched → ${title}`
    },
    btw: {
      answering: (taskId: string) => `btw ${taskId} — answering from a conversation snapshot`,
      usage: '/btw <question>'
    },
    busy: {
      mode: (mode: string) => `busy input mode: ${mode}`,
      usage: 'usage: /busy [queue|steer|interrupt|status]'
    },
    compress: {
      // {0}=message count (pre-formatted)
      compressedOne: (count: string) => `compressed ${count} message`,
      compressedOther: (count: string) => `compressed ${count} messages`,
      nothing: 'nothing to compress',
      // {0}=compact token count, appended to compressedOne/Other
      tokSuffix: (tokens: string) => ` · ${tokens} tok`
    },
    fast: {
      mode: (mode: string) => `fast mode: ${mode}`,
      usage: 'usage: /fast [normal|fast|status|on|off|toggle]'
    },
    indicator: {
      current: (style: string) => `indicator: ${style}`,
      switched: (style: string) => `indicator → ${style}`,
      usage: (styles: string) => `usage: /indicator [${styles}]`
    },
    model: {
      cancel: 'Cancel',
      expensiveDetail: 'This model has unusually high known pricing.',
      expensiveTitle: 'Expensive model selection',
      invalidResponse: 'error: invalid response: model switch',
      switchAnyway: 'Switch anyway',
      switched: (model: string) => `model → ${model}`,
      switchedDeferred: (model: string) => `model → ${model} (applies next turn)`
    },
    personality: {
      changed: (value: string) => `personality: ${value}`,
      changedCleared: (value: string) => `personality: ${value} · transcript cleared`,
      defaultValue: 'default'
    },
    pet: {
      noOutput: '/pet: no output',
      // {0}=warning text, {1}=command output
      warning: (warning: string, body: string) => `warning: ${warning}\n${body}`
    },
    reasoning: {
      current: (value: string) => `reasoning: ${value}`,
      // {0}=effort value, {1}=display mode
      currentWithDisplay: (value: string, display: string) => `reasoning: ${value} · display ${display}`
    },
    sessions: {
      busyGuardAction: 'switch sessions'
    },
    skin: {
      current: (skin: string) => `skin: ${skin}`,
      defaultValue: 'default',
      switched: (skin: string) => `skin → ${skin}`
    },
    theme: {
      current: (theme: string) => `theme: ${theme}`,
      switched: (theme: string) => `theme → ${theme}`,
      usage: 'usage: /theme [auto|light|dark]'
    },
    usage: {
      balanceTitle: 'Balance',
      compressions: (count: string) => `Compressions: ${count}`,
      // {0}=used tokens (with ~ mark when estimated), {1}=max tokens, {2}=percent (with ~ mark)
      context: (used: string, max: string, percent: string) => `Context: ${used} / ${max} (${percent}%)`,
      cta: 'Run /subscription to change plan · /topup to add to your balance',
      freeNote: '> Free · free models only. Run /subscription to reach paid models.',
      freePlan: 'Free',
      lowBalanceFallback: 'under $5',
      lowNote: (left: string) => `! Low balance · ${left} left. Run /topup or /subscription.`,
      noCalls: 'no API calls yet',
      nousBalanceTitle: 'Nous balance',
      plan: (plan: string) => `Plan: ${plan}`,
      // {0}=plan name, {1}=renewal date display
      planRenews: (plan: string, renews: string) => `Plan: ${plan} · renews ${renews}`,
      rowApiCalls: 'API calls',
      rowInputTokens: 'Input tokens',
      rowModel: 'Model',
      rowOutputTokens: 'Output tokens',
      rowTotalTokens: 'Total tokens',
      usageTitle: 'Usage'
    },
    verbose: {
      current: (value: string) => `verbose: ${value}`
    },
    voice: {
      disabled: 'Voice mode disabled.',
      enabled: 'Voice mode enabled',
      enabledWithTts: 'Voice mode enabled (TTS enabled)',
      off: 'OFF',
      offHint: '  /voice off  to disable voice mode',
      on: 'ON',
      recordHint: (recordKey: string) => `  ${recordKey} to start/stop recording`,
      requirements: '  Requirements:',
      statusMode: (mode: string) => `  Mode:       ${mode}`,
      statusRecordKey: (recordKey: string) => `  Record key: ${recordKey}`,
      statusTitle: 'Voice Mode Status',
      statusTts: (tts: string) => `  TTS:        ${tts}`,
      ttsDisabled: 'Voice TTS disabled.',
      ttsEnabled: 'Voice TTS enabled.',
      ttsHint: '  /voice tts  to toggle speech output'
    },
    yolo: {
      off: 'yolo off',
      on: 'yolo on'
    }
  },
  subscription: {
    billingUnreachable: 'Could not reach the billing service — check your connection, then retry.',
    manageUrlFailed: 'Could not build manage URL — is your portal configured?',
    notLoggedIn: 'Not logged into Nous Portal — run /portal to log in, then /subscription.',
    openBrowserFailedManage: (url: string) =>
      `Could not open browser — visit your subscription page manually at ${url}`,
    openBrowserFailedPortal: (url: string) => `Could not open browser — visit ${url} to finish.`,
    openingManage: 'Opening your subscription page in the browser — finish there, then re-run /subscription.',
    openingPortal: 'Opening the portal in your browser — finish there, then re-run /subscription.'
  },
  topup: {
    charge: {
      authenticationRequired:
        '🔴 Your bank requires verification (3DS). Complete it on the portal to finish this purchase.',
      cardDeclined: '🔴 Your card was declined. Try another card on the portal.',
      couldNotCheck: (reason: string) => `🔴 Could not check the charge: ${reason}`,
      couldNotCheckFallback: 'error',
      creditsFallback: 'Credits',
      failedReason: (reason: string) => `🔴 The charge didn't go through (${reason}).`,
      paymentMethodExpired: '🔴 Your card has expired. Update it on the portal.',
      // {0}=pre-formatted dollar amount ("$100") or creditsFallback
      settled: (amount: string) => `✅ ${amount} added.`,
      submitted: '💳 Charge submitted — confirming settlement…',
      timedOut:
        '🟡 Still processing after 5 minutes — this is a timeout, not a failure. Check /topup or the portal shortly.',
      unconfirmed: '🟡 Your last charge’s outcome is unconfirmed — check your balance/history before retrying.'
    },
    error: {
      autoTopUpDisabledFailures:
        'Auto-reload was turned off after repeated charge failures. Fix the card issue, then re-enable it from /topup → Auto-reload.',
      consentRequired:
        'This action needs a one-time card confirmation and consent step on the portal before it can proceed.',
      generic: (message: string) => `🔴 ${message}`,
      genericFallback: 'Billing request failed.',
      idempotencyConflict: '🔴 That charge key was already used for a different amount. Start a fresh top-up.',
      insufficientScope: 'This needs Remote Spending allowed. Start a top-up to allow it, then retry.',
      monthlyCapExceeded: '🔴 Monthly spend cap reached.',
      // {0}=remaining USD headroom (number as sent by the server)
      monthlyCapExceededRemaining: (remaining: string) => `🔴 Monthly spend cap reached — $${remaining} headroom left.`,
      noPaymentMethod:
        "💳 No saved card for terminal charges yet. Set one up on the portal (one-time credit buys don't save a reusable card).",
      orgAccessDenied:
        "This token isn't bound to an org you can manage. Sign in with the right org, or manage this on the portal.",
      portal: (url: string) => `Portal: ${url}`,
      // {0}=retry suffix from retryIn (or '')
      rateLimited: (retrySuffix: string) =>
        `🟡 Too many charges right now${retrySuffix}. This isn't a payment failure.`,
      remoteSpendingDisabled:
        "Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page.",
      retryIn: (minutes: string) => ` (try again in ~${minutes} min)`,
      revokedByAdmin: 'An admin stopped remote spending for this terminal.',
      revokedByYou: 'You stopped remote spending for this terminal.',
      // {0}=who-revoked sentence (revokedByAdmin / revokedByYou)
      revokedReconnect: (who: string) => `${who} Reconnect to restore — run /portal to re-authorize this terminal.`,
      roleRequired:
        'Adding funds needs someone with billing permissions (owner, admin, or finance admin), or manage this on the portal.',
      sessionRevoked: 'Your session was logged out. Run /portal to log in again.',
      // {0}=retry suffix from retryIn (or '')
      stripeUnavailable: (retrySuffix: string) =>
        `🟡 Stripe is having trouble right now — try again shortly${retrySuffix}.`,
      upgradeCapExceeded:
        '🔴 Daily plan-change limit reached (5 per org) — try again tomorrow, or manage this on the portal.'
    },
    notLoggedIn: '💳 Not logged into Nous Portal — run /portal to log in, then /topup.',
    openingPortal: (url: string) => `Opening portal: ${url}`,
    validate: {
      invalid: 'Enter a dollar amount, e.g. 100 (max 2 decimal places).',
      maximum: (max: string) => `Maximum is $${max}.`,
      minimum: (min: string) => `Minimum is $${min}.`,
      notPositive: 'Amount must be greater than $0.'
    }
  }
}
