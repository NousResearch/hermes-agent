// app/userMessages.ts copy: gateway/transport failure wording, turn-failure
// titles + next steps, withdrawn-prompt notices, empty states.
// Owned namespace: `userMessages`.
//
// Every slash command cited here exists in ui-tui/src/app/slash/commands
// (/logs, /retry, /model, /update, /resume, /sessions, /quit, /compress,
// /skills, /setup) and `hermes doctor` is a real subcommand — keep them as-is.

export const userMessagesEn = {
  userMessages: {
    /** `{0}` = the (already clipped) raw detail text. */
    details: (text: string) => `Details: ${text}`,

    backend: {
      restarting:
        'Hermes stopped unexpectedly — restarting and reopening your chat (the reply in progress was lost).',
      restartingActivity: 'Hermes stopped unexpectedly · restarting…',
      connectionLost: 'Connection to Hermes lost — reconnecting and reopening your chat…',
      connectionLostActivity: 'connection lost · reconnecting…',
      gaveUpTitle: 'Hermes stopped and could not be restarted. Your chat is saved.',
      /** `{0}` = process exit code. */
      gaveUpTitleWithCode: (code: string) =>
        `Hermes stopped (exit code ${code}) and could not be restarted. Your chat is saved.`,
      gaveUpReconnect:
        'Hermes keeps trying to reconnect in the background and reopens this chat when it succeeds; if it does not, type /resume.',
      gaveUpLogs: 'Type /logs for the full log, or /quit and run `hermes doctor` to check the install.',
      gaveUpActivity: 'Hermes stopped · /logs for details',
      /** `{0}` = seconds until the next attempt. */
      reconnecting: (secs: string) => `retrying in ${secs}s`,
      /** `{0}` = seconds until the next attempt, `{1}` = attempt number. */
      reconnectingAttempt: (secs: string, attempt: string) => `retrying in ${secs}s (attempt ${attempt})`,
      slowStart:
        'Hermes is taking longer than usual to start. Still waiting… If it never connects: /logs shows the last backend output; /quit and run `hermes doctor`.',
      slowStartStatus: 'still starting…',
      stderrProblem: 'Something went wrong inside Hermes · /logs for details',
      /** `{0}` = the Python exception class name (e.g. `ValueError`). */
      stderrProblemNamed: (what: string) => `Something went wrong inside Hermes (${what}) · /logs for details`
    },

    rpc: {
      versionSkew:
        'The terminal UI and the Hermes backend are out of sync (different versions). Run /update, or exit and run `hermes update`, then start the TUI again.',
      sessionNotFound:
        'This chat is no longer attached to the backend (it was idle or the backend restarted). Your history is saved: type /resume to reopen it.',
      notConnected:
        'Hermes is not connected right now, so that was not sent. It reconnects automatically; wait a moment and try again, or type /logs if this persists.',
      /** `{0}` = timeout in seconds. */
      timedOut: (secs: string) =>
        `Hermes did not answer within ${secs}s. Try again; if it keeps happening, type /logs and report the last lines.`,
      /** `{0}` = slash command name without the slash. */
      slashTimedOut: (command: string) =>
        `/${command} did not finish: the command helper timed out. Try again; if it keeps happening, type /logs and report the last lines.`,
      /** `{0}` = slash command name without the slash. */
      slashCrashed: (command: string) =>
        `/${command} did not finish: the command helper crashed. Try again; type /logs for the trace.`
    },

    turn: {
      /** `{0}` = failure title (possibly with provider suffix). */
      notAnswered: (title: string) => `${title}. Your message was not answered.`,
      /** `{0}` = failure title, `{1}` = provider id. */
      withProvider: (title: string, provider: string) => `${title} (${provider})`,
      // `hintNoRetry` is the next step when error_surface.retryable is false —
      // it replaces the "/retry" advice with picking another model.
      code: {
        auth: { title: 'The model provider rejected the API key', hint: 'Fix the key with /model, then /retry.' },
        authPermanent: {
          title: 'The model provider rejected the API key',
          hint: 'Fix the key with /model, then /retry.'
        },
        billing: {
          title: 'The model provider reports no credit left',
          hint: 'Top up the account or switch with /model.'
        },
        billingUnverified: {
          title: 'The model provider reports no credit left',
          hint: 'Top up the account or switch with /model.'
        },
        contentPolicyBlocked: {
          title: 'The model provider refused this request (content policy)',
          hint: 'Rephrase and send again.'
        },
        contextOverflow: { title: 'The conversation is too long for this model', hint: 'Run /compress, then /retry.' },
        formatError: {
          title: 'The model provider rejected the request format',
          hint: 'Try /retry; if it persists, switch with /model.',
          hintNoRetry: 'Pick another model with /model; if it persists, switch with /model.'
        },
        modelNotFound: { title: 'The model provider does not know this model', hint: 'Pick another model with /model.' },
        overloaded: { title: 'The model provider is overloaded', hint: 'Wait a moment, then /retry.' },
        payloadTooLarge: { title: 'The request was too large for this model', hint: 'Run /compress, then /retry.' },
        providerPolicyBlocked: {
          title: 'The model provider refused this request (account policy)',
          hint: 'Switch with /model.'
        },
        rateLimit: { title: 'The model provider is rate-limiting requests', hint: 'Wait a moment, then /retry.' },
        serverError: { title: 'The model provider had an internal error', hint: 'Wait a moment, then /retry.' },
        sslCertVerification: {
          title: 'The connection to the model provider could not be verified (TLS)',
          hint: "Check the endpoint's certificate, then /retry."
        },
        timeout: {
          title: 'The model provider did not answer in time',
          hint: 'Try /retry; if it keeps happening, switch with /model.',
          hintNoRetry: 'Pick another model with /model; if it keeps happening, switch with /model.'
        },
        upstreamBlocked: {
          title: 'A firewall/CDN in front of the model provider blocked the request',
          hint: "Set a User-Agent via the provider's extra_headers, or switch with /model."
        },
        upstreamRateLimit: {
          title: 'The model provider is rate-limiting requests',
          hint: 'Wait a moment, then /retry.'
        }
      },
      layer: {
        auth: { title: 'The model provider rejected the credentials', hint: 'Fix them with /model, then /retry.' },
        billing: {
          title: 'The model provider reports no credit left',
          hint: 'Top up the account or switch with /model.'
        },
        disk: { title: 'The disk is full, so Hermes could not save the turn', hint: 'Free some space, then /retry.' },
        endpoint: {
          title: 'Your custom model endpoint did not answer',
          hint: 'Check the endpoint is running, then /retry.'
        },
        gateway: {
          title: 'Hermes hit an internal error while running this turn',
          hint: 'Send /retry; type /logs for the trace.',
          hintNoRetry: 'Pick another model with /model; type /logs for the trace.'
        },
        provider: {
          title: 'The model provider returned an error',
          hint: 'Send /retry, or switch with /model.',
          hintNoRetry: 'Pick another model with /model, or switch with /model.'
        },
        streaming: {
          title: 'The connection to the model provider dropped mid-reply',
          hint: 'Send /retry.',
          hintNoRetry: 'Pick another model with /model.'
        }
      },
      fallback: {
        title: 'The request failed',
        hint: 'Send /retry, or switch with /model.',
        hintNoRetry: 'Pick another model with /model, or switch with /model.'
      }
    },

    promptTimeout: {
      secret:
        'Secret prompt closed: no answer in time, so the step that needed it was skipped. Send your request again when you are ready to enter it.',
      sudo: 'Password prompt closed: no answer in time, so the command was skipped. Send your request again when you are ready to enter it.',
      vaultCode:
        'Verification-code prompt closed: no answer in time, so the sign-in was skipped. Send your request again when you have the code.',
      vaultSaveLogin:
        'Save-login prompt closed: no answer in time, so nothing was saved. Send your request again when you are ready.',
      vaultUnlockPrompt:
        'Unlock prompt closed: no answer in time, so the password manager stayed locked. Send your request again when you are ready to unlock it.'
    },

    credential: {
      currentProvider: 'the current provider',
      /** `{0}` = provider id (or `currentProvider`), used twice in the sentence. */
      missingKey: (provider: string) =>
        `No API key is set for ${provider}, so messages will fail. Type /model, pick ${provider}, and paste a key (or run /setup).`
    },

    skills: {
      noneInstalled: 'No skills installed yet. Type /skills browse to see the catalog, or /skills install <name>.'
    }
  }
}
