// Shared-gateway (canonical authority) client text: durable submission/queue receipts, image
// staging, launch-contract refusals and the slash commands mapped onto `session.mutate`.
// Owned namespace: `canonical`. Function leaves take positional args; packs use `{0}`, `{1}`.
// Status-bar phrases here are free text (nothing compares against them).

export const canonicalEn = {
  canonical: {
    // app/slash/canonicalSessionControls.ts + canonicalSessionCommands.ts
    controls: {
      modelUsage: 'usage: /model <model> [--provider <provider>] [--session]',
      unsupportedModelOption: (flag: string) => `unsupported canonical model option: ${flag}`,
      identityUnavailable: 'session execution identity unavailable; reconnect before editing',
      compressedStale: 'compressed transcript is stale; reopen the session',
      compressed: '✓ transcript compressed',
      notAvailable: (command: string) => `/${command} is not available on the shared gateway yet`,
      prefNotSaved: (command: string) => `/${command} applies to this view only; not saved on the shared gateway`,
      launchToolsets: (toolsets: string) => `toolsets fixed at session launch: ${toolsets}`,
      usageLastTurnTitle: 'Usage · last turn',
      usageNoTurn: 'no completed turn in this view yet',
      usageTotalsUnavailable: 'session totals are not available on the shared gateway yet',
      rowCost: 'Estimated cost'
    },
    // app/submissionCore.ts — durable admission receipts.
    submit: {
      correction: (status: string) => `correction ${status}`,
      correctionRejected: 'correction rejected — input retained',
      admissionNotConfirmed: 'admission not confirmed — input retained; use Alt+K to retry with the same identity',
      statusAdmissionUnconfirmed: 'admission unconfirmed',
      executionUnconfirmed: (message: string) => `execution status unconfirmed: ${message}`,
      legacyDelivery: 'durable admission unavailable for this session — using legacy delivery',
      legacyUnconfirmed: 'legacy delivery unconfirmed — input retained; check the session before sending a new input',
      legacyUnconfirmedWith: (message: string) =>
        `legacy delivery unconfirmed: ${message} — input retained; check the session before sending a new input`,
      statusDeliveryUnconfirmed: 'delivery unconfirmed',
      deliveryNotResent: 'delivery unconfirmed — check the session before sending a new input; retained input was not resent',
      busyTextOnly: 'busy corrections accept text only — image input retained; submit it with /queue',
      inputRetained: (message: string) => `input retained: ${message} — Alt+K retries the same submission`,
      imageNotSubmitted: (message: string) => `image not submitted: ${message} — input retained`,
      statusImageNotSubmitted: 'image not submitted',
      interruptFailed: (message: string) => `interrupt failed: ${message}`,
      startupQuerySwitched: 'startup query skipped: active session changed'
    },
    // hooks/useQueue.ts, app/useSubmission.ts, app/useComposerState.ts — the local input journal.
    queue: {
      unconfirmedPrefix: '[unconfirmed · Alt+K retry] ',
      // {0}=admission status identifier (queued/started/unknown), {1}=input text
      serverRow: (status: string, text: string) => `[${status}] ${text}`,
      discardFailed: (message: string) => `discard failed: ${message}`,
      journalCleanupFailed: (message: string) => `input delivered; journal cleanup failed: ${message}`,
      notSavedQueueKept: (message: string) => `input not saved: ${message} — queue kept`,
      notSavedDraftKept: (message: string) => `input not saved: ${message} — draft kept`,
      statusInputNotSaved: 'input not saved',
      unknownExecution: 'unknown execution — Ctrl+X to discard before retrying',
      invalidRecord: (name: string) => `Invalid pending input record: ${name}`,
      clipboardImageFailed: (message: string) => `clipboard image failed: ${message}`,
      noClipboardImage: 'No image found in clipboard'
    },
    // lib/imageAttachments.ts — owner-staged image validation.
    images: {
      unsupported: 'Unsupported image: expected PNG, JPEG, GIF, or WebP bytes',
      tooLarge: 'Image exceeds 20 MiB upload limit',
      notFound: (path: string) => `Image not found: ${path}`,
      noOwnerPath: 'Image upload did not return an owner path'
    },
    // canonicalGateway.ts — launch-contract refusals; components/branding.tsx.
    launch: {
      noTuiPolicy: 'gateway does not support tui session policy; update/restart the gateway',
      unsupportedOptions: (options: string) => `gateway does not support TUI launch options: ${options}`,
      invalidToolProgress: (value: string) => `invalid HERMES_TUI_TOOL_PROGRESS: ${value}`,
      inventoryUnavailable: 'Tool and skill inventory is not exposed by this runtime.'
    }
  }
}
