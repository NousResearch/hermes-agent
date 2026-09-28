// User-facing wording for gateway/transport failures in the TUI. Pure functions
// so the copy — and the "what happened / what to do" shape — is unit-testable
// without rendering. Every slash command cited here exists in
// ui-tui/src/app/slash/commands (/logs, /retry, /model, /update, /resume,
// /sessions, /quit) and `hermes doctor` is a real subcommand.

import type { ErrorSurface } from '@hermes/shared/gateway-events'

import { type Locale, translate, type TranslationKey } from '../i18n/index.js'

/** JSON-RPC error codes the gateway answers with. */
export const RPC_INVALID_PARAMS = 4000
export const RPC_SESSION_NOT_FOUND = 4001
export const RPC_NOT_DISPATCHABLE = 4018
export const RPC_UNKNOWN_METHOD = -32601

const DETAIL_LIMIT = 300

interface RpcErrorShape {
  code?: number
  message?: string
}

const rpcShape = (err: unknown): RpcErrorShape =>
  err instanceof Error ? { code: (err as { code?: number }).code, message: err.message } : {}

const detailLine = (raw: string | undefined, locale: Locale = 'en'): string | null => {
  const text = (raw ?? '').replace(/\s+/g, ' ').trim()

  if (!text) {
    return null
  }

  return translate(locale, 'messages.detail', {
    detail: text.length > DETAIL_LIMIT ? `${text.slice(0, DETAIL_LIMIT - 1)}…` : text
  })
}

// ── Backend process lifecycle ─────────────────────────────────────────────

export const BACKEND_RESTARTING = (locale: Locale = 'en') => translate(locale, 'messages.backend_restarting')

export const BACKEND_RESTARTING_ACTIVITY = (locale: Locale = 'en') =>
  translate(locale, 'messages.backend_restarting_activity')

// Attached (dashboard / embedded) mode: only the socket dropped; Hermes and any
// reply in progress are still alive on the backend and come back on reconnect.
export const CONNECTION_LOST = (locale: Locale = 'en') => translate(locale, 'messages.connection_lost')

export const CONNECTION_LOST_ACTIVITY = (locale: Locale = 'en') =>
  translate(locale, 'messages.connection_lost_activity')

export const backendGaveUp = (code: null | number, lastLine?: string, locale: Locale = 'en'): string => {
  const exit = code === null ? '' : translate(locale, 'messages.exit', { code: code ?? 0 })
  const detail = detailLine(lastLine, locale)

  return [
    translate(locale, 'messages.gaveUp', { exit }),
    detail,
    translate(locale, 'messages.saved'),
    translate(locale, 'messages.doctor')
  ]
    .filter(Boolean)
    .join('\n')
}

export const BACKEND_GAVE_UP_ACTIVITY = (locale: Locale = 'en') =>
  translate(locale, 'messages.backend_gave_up_activity')

/** Last line of the backend log tail that is not our own [lifecycle]/[startup] bookkeeping. */
export const lastStderrLine = (tail: string): string | undefined =>
  tail
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !/^\[(?:lifecycle|startup|protocol|sidecar|spawn)\]/.test(l))
    .at(-1)

export const backendReconnecting = (
  attempt: number | undefined,
  delayMs: number | undefined,
  locale: Locale = 'en'
): string => {
  const secs = Math.max(1, Math.round((delayMs ?? 1000) / 1000))
  const n = attempt && attempt > 0 ? translate(locale, 'messages.attempt', { attempt }) : ''

  return translate(locale, 'messages.reconnecting', { secs, attempt: n })
}

export const BACKEND_SLOW_START = (locale: Locale = 'en') => translate(locale, 'messages.backend_slow_start')

export const BACKEND_SLOW_START_STATUS = (locale: Locale = 'en') =>
  translate(locale, 'messages.backend_slow_start_status')

// ── stderr noise ──────────────────────────────────────────────────────────

// Only real failures: a traceback, a CRITICAL log line, an `XxxError:` / `XxxException:`
// head, or the gateway's own turn/exit markers. Dependency `DeprecationWarning` /
// `UserWarning` lines are noise and stay in /logs only.
const STDERR_PROBLEM_RE = /Traceback|\b[A-Z][A-Za-z]*(?:Error|Exception)\b:|CRITICAL|\[gateway-turn\]|\[gateway-exit\]/

/** Only lines that look like a failure earn an activity row; the rest stay in /logs. */
export const stderrLooksLikeProblem = (line: string): boolean => STDERR_PROBLEM_RE.test(line)

export const stderrProblemActivity = (line: string, locale: Locale = 'en'): string => {
  const m = /([A-Z][A-Za-z]*(?:Error|Exception)):/.exec(line)
  const what = m ? ` (${m[1]})` : ''

  return translate(locale, 'messages.stderr', { what })
}

// ── RPC errors ────────────────────────────────────────────────────────────

const VERSION_SKEW_RE = /Extra inputs are not permitted|^unknown method:/

/** The Ink bundle and the Python backend disagree on the wire: stale dist or an older attached backend. */
export const isVersionSkewError = (err: unknown): boolean => {
  const { code, message } = rpcShape(err)

  return (
    code === RPC_UNKNOWN_METHOD ||
    (code === RPC_INVALID_PARAMS && VERSION_SKEW_RE.test(message ?? '')) ||
    (code === undefined && VERSION_SKEW_RE.test(message ?? ''))
  )
}

export const VERSION_SKEW_MESSAGE = (locale: Locale = 'en') => translate(locale, 'messages.version_skew_message')

const SESSION_NOT_FOUND_RE = /session not found/i
const NOT_CONNECTED_RE = /^gateway not (?:connected|running)\b/
const TIMED_OUT_RE = /^request timed out after (\d+)s/

type RpcErrorRow = [
  matcher: (code: number | undefined, text: string) => RegExpExecArray | boolean | null,
  render: (m: RegExpExecArray | null, locale: Locale) => string
]

// Ordered: first matching row wins. 4001 is reused by the backend for unrelated
// refusals ("no active session", "slug is required", NOT_OWNER), so the code
// alone must not trigger the /resume copy — only the "session not found" text.
const RPC_ERROR_ROWS: RpcErrorRow[] = [
  [
    (code, text) => (code === RPC_SESSION_NOT_FOUND || code === undefined) && SESSION_NOT_FOUND_RE.test(text),
    (_m, locale) => translate(locale, 'messages.sessionMissing')
  ],
  [(_code, text) => NOT_CONNECTED_RE.test(text), (_m, locale) => translate(locale, 'messages.disconnected')],
  [
    (_code, text) => TIMED_OUT_RE.exec(text),
    (m, locale) => translate(locale, 'messages.timeout', { secs: m?.[1] ?? '?' })
  ]
]

let rpcErrorLogSink: ((line: string) => void) | null = null

/** Where describeRpcError records the raw wire text it replaced (the /logs buffer). */
export const setRpcErrorLogSink = (sink: ((line: string) => void) | null): void => {
  rpcErrorLogSink = sink
}

const logReplacedWireText = (code: number | undefined, text: string): void => {
  rpcErrorLogSink?.(`[rpc] ${code === undefined ? '' : `code=${code} `}${text}`)
}

/** Rewrite transport/session errors into plain words; other errors pass through. */
export const describeRpcError = (err: unknown, locale: Locale = 'en'): string => {
  const { code, message } = rpcShape(err)
  const text = message ?? (typeof err === 'string' && err.trim() ? err : translate(locale, 'messages.requestFailed'))

  if (isVersionSkewError(err)) {
    logReplacedWireText(code, text)

    return VERSION_SKEW_MESSAGE(locale)
  }

  for (const [matcher, render] of RPC_ERROR_ROWS) {
    const m = matcher(code, text)

    if (m) {
      logReplacedWireText(code, text)

      return render(m === true ? null : m, locale)
    }
  }

  return text
}

/** The slash worker (built-in command helper) failed; name the command, not the helper. */
export const describeSlashExecError = (command: string, err: unknown, locale: Locale = 'en'): string => {
  const { message } = rpcShape(err)
  const text = message ?? ''

  if (/slash worker timed out/.test(text)) {
    return translate(locale, 'messages.slashTimeout', { command })
  }

  if (/slash worker (?:exited|closed pipe|start failed)/.test(text)) {
    const detail = detailLine(text.replace(/^slash worker (?:exited|closed pipe:?|start failed:?)\s*/, ''), locale)

    return [translate(locale, 'messages.slashCrashed', { command }), detail].filter(Boolean).join('\n')
  }

  return describeRpcError(err, locale)
}

// slash.exec answers 4018 with exactly these texts when it does NOT own the
// command (tui_gateway/methods_tools.py). Every other 4018 came from a
// command.dispatch handler slash.exec already forwarded to (/retry, /undo,
// /compress, /queue, bundles): re-dispatching would run a mutating command twice.
const NOT_MINE_REFUSAL_RE = /^skill command: use command\.dispatch for \/|use command\.dispatch for \/snapshot restore/

/** command.dispatch is only a fallback for "slash.exec does not own this command" refusals. */
export const shouldFallbackToDispatch = (err: unknown): boolean => {
  const { code, message } = rpcShape(err)

  if (code === RPC_NOT_DISPATCHABLE) {
    return NOT_MINE_REFUSAL_RE.test(message ?? '')
  }

  if (code !== undefined) {
    return false
  }

  // Legacy/attached backends without a code: keep the historical behaviour
  // unless the text is unmistakably a helper failure.
  return !/slash worker|timed out|not connected|not running/.test(message ?? '')
}

// ── Turn failures (message.complete status=error) ─────────────────────────

const TURN_CODE_COPY: Record<string, [TranslationKey, TranslationKey]> = {
  auth: ['failure.theModelProviderRejectedTheApiKey', 'failure.fixTheKeyWithModelThenRetry'],
  auth_permanent: ['failure.theModelProviderRejectedTheApiKey', 'failure.fixTheKeyWithModelThenRetry'],
  billing: ['failure.theModelProviderReportsNoCreditLeft', 'failure.topUpTheAccountOrSwitchWithModel'],
  billing_unverified: ['failure.theModelProviderReportsNoCreditLeft', 'failure.topUpTheAccountOrSwitchWithModel'],
  content_policy_blocked: ['failure.theModelProviderRefusedThisRequestContentPolicy', 'failure.rephraseAndSendAgain'],
  context_overflow: ['failure.theConversationIsTooLongForThisModel', 'failure.runCompressThenRetry'],
  format_error: ['failure.theModelProviderRejectedTheRequestFormat', 'failure.tryRetryIfItPersistsSwitchWithModel'],
  model_not_found: ['failure.theModelProviderDoesNotKnowThisModel', 'failure.pickAnotherModelWithModel'],
  overloaded: ['failure.theModelProviderIsOverloaded', 'failure.waitAMomentThenRetry'],
  payload_too_large: ['failure.theRequestWasTooLargeForThisModel', 'failure.runCompressThenRetry'],
  provider_policy_blocked: ['failure.theModelProviderRefusedThisRequestAccountPolicy', 'failure.switchWithModel'],
  rate_limit: ['failure.theModelProviderIsRateLimitingRequests', 'failure.waitAMomentThenRetry'],
  server_error: ['failure.theModelProviderHadAnInternalError', 'failure.waitAMomentThenRetry'],
  ssl_cert_verification: [
    'failure.theConnectionToTheModelProviderCouldNotBeVerifiedTls',
    'failure.checkTheEndpointSCertificateThenRetry'
  ],
  timeout: ['failure.theModelProviderDidNotAnswerInTime', 'failure.tryRetryIfItKeepsHappeningSwitchWithModel'],
  upstream_blocked: [
    'failure.aFirewallCdnInFrontOfTheModelProviderBlockedTheRequest',
    'failure.setAUserAgentViaTheProviderSExtraHeadersOrSwitchWithModel'
  ],
  upstream_rate_limit: ['failure.theModelProviderIsRateLimitingRequests', 'failure.waitAMomentThenRetry']
}

const TURN_LAYER_COPY: Record<string, [TranslationKey, TranslationKey]> = {
  auth: ['failure.theModelProviderRejectedTheCredentials', 'failure.fixThemWithModelThenRetry'],
  billing: ['failure.theModelProviderReportsNoCreditLeft', 'failure.topUpTheAccountOrSwitchWithModel'],
  disk: ['failure.theDiskIsFullSoHermesCouldNotSaveTheTurn', 'failure.freeSomeSpaceThenRetry'],
  endpoint: ['failure.yourCustomModelEndpointDidNotAnswer', 'failure.checkTheEndpointIsRunningThenRetry'],
  gateway: ['failure.hermesHitAnInternalErrorWhileRunningThisTurn', 'failure.sendRetryTypeLogsForTheTrace'],
  provider: ['failure.theModelProviderReturnedAnError', 'failure.sendRetryOrSwitchWithModel'],
  streaming: ['failure.theConnectionToTheModelProviderDroppedMidReply', 'failure.sendRetry']
}

const TURN_DEFAULT_COPY: [TranslationKey, TranslationKey] = [
  'failure.theRequestFailed',
  'failure.sendRetryOrSwitchWithModel'
]

export interface TurnFailure {
  error?: null | string
  error_surface?: ErrorSurface | null | Record<string, unknown>
  recoverable?: boolean | null
}

/** Plain title + dimmed detail + next step for a failed turn with no reply text. */
export const describeTurnFailure = (payload: TurnFailure, locale: Locale = 'en'): string => {
  const surface = (payload.error_surface ?? {}) as { code?: unknown; layer?: unknown; provider?: unknown }
  const code = typeof surface.code === 'string' ? surface.code : ''
  const layer = typeof surface.layer === 'string' ? surface.layer : ''
  const provider = typeof surface.provider === 'string' && surface.provider ? ` (${surface.provider})` : ''
  const [titleKey, hintKey] = TURN_CODE_COPY[code] ?? TURN_LAYER_COPY[layer] ?? TURN_DEFAULT_COPY
  // The backend always sets recoverable=true on a turn error; error_surface.retryable
  // is the signal that actually says whether /retry can help.
  const retryable = (surface as { retryable?: unknown }).retryable !== false && payload.recoverable !== false
  const title = translate(locale, titleKey)
  const nextStep = translate(
    locale,
    !retryable &&
      [
        'failure.tryRetryIfItPersistsSwitchWithModel',
        'failure.tryRetryIfItKeepsHappeningSwitchWithModel',
        'failure.sendRetryTypeLogsForTheTrace',
        'failure.sendRetryOrSwitchWithModel',
        'failure.sendRetry'
      ].includes(hintKey)
      ? 'failure.pickAnotherModelWithModel'
      : hintKey
  )
  const raw = (payload.error ?? '').replace(/^Error:\s*/, '')

  return [translate(locale, 'messages.unanswered', { title, provider }), detailLine(raw, locale), nextStep]
    .filter(Boolean)
    .join('\n')
}

/** True when the assistant slot carries nothing but the backend's "Error: …" fallback text. */
export const isBareErrorText = (text: string, error: null | string | undefined): boolean => {
  const t = text.trim()

  return !t || t === `Error: ${error ?? ''}`.trim() || t === (error ?? '').trim()
}

// ── Withdrawn password / secret prompts ───────────────────────────────────

const PROMPT_TIMEOUT_COPY: Record<string, TranslationKey> = {
  secret: 'timeout.secret',
  sudo: 'timeout.sudo',
  'vault.code': 'timeout.vault.code',
  'vault.save_login': 'timeout.vault.save_login',
  'vault.unlock_prompt': 'timeout.vault.unlock_prompt'
}

export const promptTimeoutNotice = (
  method: string | undefined,
  reason: string | undefined,
  locale: Locale = 'en'
): null | string =>
  reason === 'timeout' && method && PROMPT_TIMEOUT_COPY[method] ? translate(locale, PROMPT_TIMEOUT_COPY[method]) : null

// ── session.info warnings ─────────────────────────────────────────────────

const MISSING_KEY_RE = /^No API key configured for provider '([^']*)'/

/** The backend's credential warning names the break; add the fix (/model saves a key in place). */
export const describeCredentialWarning = (warning: string, locale: Locale = 'en'): string => {
  const m = MISSING_KEY_RE.exec(warning)

  if (!m) {
    return warning
  }

  const provider = m[1] || translate(locale, 'messages.currentProvider')

  return translate(locale, 'messages.missingKey', { provider })
}
