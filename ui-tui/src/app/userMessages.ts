// User-facing wording for gateway/transport failures in the TUI. Pure functions
// so the copy — and the "what happened / what to do" shape — is unit-testable
// without rendering. The English copy lives in i18n/en/userMessages.ts
// (namespace `userMessages`); every exported message is a function so it is
// resolved against the active locale at call time, never at import time.

import type { ErrorSurface } from '@hermes/shared/gateway-events'

import { messages, t } from '../i18n/runtime.js'
import type { Translations } from '../i18n/types.js'

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

const detailLine = (raw: string | undefined): string | null => {
  const text = (raw ?? '').replace(/\s+/g, ' ').trim()

  if (!text) {
    return null
  }

  return t('userMessages.details', text.length > DETAIL_LIMIT ? `${text.slice(0, DETAIL_LIMIT - 1)}…` : text)
}

// ── Backend process lifecycle ─────────────────────────────────────────────

export const backendRestarting = (): string => t('userMessages.backend.restarting')

export const backendRestartingActivity = (): string => t('userMessages.backend.restartingActivity')

// Attached (dashboard / embedded) mode: only the socket dropped; Hermes and any
// reply in progress are still alive on the backend and come back on reconnect.
export const connectionLost = (): string => t('userMessages.backend.connectionLost')

export const connectionLostActivity = (): string => t('userMessages.backend.connectionLostActivity')

export const backendGaveUp = (code: null | number, lastLine?: string): string => {
  const detail = detailLine(lastLine)

  return [
    code === null
      ? t('userMessages.backend.gaveUpTitle')
      : t('userMessages.backend.gaveUpTitleWithCode', String(code)),
    detail,
    t('userMessages.backend.gaveUpReconnect'),
    t('userMessages.backend.gaveUpLogs')
  ]
    .filter(Boolean)
    .join('\n')
}

export const backendGaveUpActivity = (): string => t('userMessages.backend.gaveUpActivity')

/** Last line of the backend log tail that is not our own [lifecycle]/[startup] bookkeeping. */
export const lastStderrLine = (tail: string): string | undefined =>
  tail
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !/^\[(?:lifecycle|startup|protocol|sidecar|spawn)\]/.test(l))
    .at(-1)

export const backendReconnecting = (attempt: number | undefined, delayMs: number | undefined): string => {
  const secs = Math.max(1, Math.round((delayMs ?? 1000) / 1000))

  return attempt && attempt > 0
    ? t('userMessages.backend.reconnectingAttempt', String(secs), String(attempt))
    : t('userMessages.backend.reconnecting', String(secs))
}

export const backendSlowStart = (): string => t('userMessages.backend.slowStart')

export const backendSlowStartStatus = (): string => t('userMessages.backend.slowStartStatus')

// ── stderr noise ──────────────────────────────────────────────────────────

// Only real failures: a traceback, a CRITICAL log line, an `XxxError:` / `XxxException:`
// head, or the gateway's own turn/exit markers. Dependency `DeprecationWarning` /
// `UserWarning` lines are noise and stay in /logs only.
const STDERR_PROBLEM_RE = /Traceback|\b[A-Z][A-Za-z]*(?:Error|Exception)\b:|CRITICAL|\[gateway-turn\]|\[gateway-exit\]/

/** Only lines that look like a failure earn an activity row; the rest stay in /logs. */
export const stderrLooksLikeProblem = (line: string): boolean => STDERR_PROBLEM_RE.test(line)

export const stderrProblemActivity = (line: string): string => {
  const m = /([A-Z][A-Za-z]*(?:Error|Exception)):/.exec(line)

  return m ? t('userMessages.backend.stderrProblemNamed', m[1]) : t('userMessages.backend.stderrProblem')
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

export const versionSkewMessage = (): string => t('userMessages.rpc.versionSkew')

const SESSION_NOT_FOUND_RE = /session not found/i
const NOT_CONNECTED_RE = /^gateway not (?:connected|running)\b/
const TIMED_OUT_RE = /^request timed out after (\d+)s/

type RpcErrorRow = [
  matcher: (code: number | undefined, text: string) => RegExpExecArray | boolean | null,
  render: (m: RegExpExecArray | null) => string
]

// Ordered: first matching row wins. 4001 is reused by the backend for unrelated
// refusals ("no active session", "slug is required", NOT_OWNER), so the code
// alone must not trigger the /resume copy — only the "session not found" text.
const RPC_ERROR_ROWS: RpcErrorRow[] = [
  [
    (code, text) => (code === RPC_SESSION_NOT_FOUND || code === undefined) && SESSION_NOT_FOUND_RE.test(text),
    () => t('userMessages.rpc.sessionNotFound')
  ],
  [
    (_code, text) => NOT_CONNECTED_RE.test(text),
    () => t('userMessages.rpc.notConnected')
  ],
  [
    (_code, text) => TIMED_OUT_RE.exec(text),
    m => t('userMessages.rpc.timedOut', m?.[1] ?? '?')
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
export const describeRpcError = (err: unknown): string => {
  const { code, message } = rpcShape(err)
  const text = message ?? (typeof err === 'string' && err.trim() ? err : t('rpc.requestFailed'))

  if (isVersionSkewError(err)) {
    logReplacedWireText(code, text)

    return versionSkewMessage()
  }

  for (const [matcher, render] of RPC_ERROR_ROWS) {
    const m = matcher(code, text)

    if (m) {
      logReplacedWireText(code, text)

      return render(m === true ? null : m)
    }
  }

  return text
}

/** The slash worker (built-in command helper) failed; name the command, not the helper. */
export const describeSlashExecError = (command: string, err: unknown): string => {
  const { message } = rpcShape(err)
  const text = message ?? ''

  if (/slash worker timed out/.test(text)) {
    return t('userMessages.rpc.slashTimedOut', command)
  }

  if (/slash worker (?:exited|closed pipe|start failed)/.test(text)) {
    const detail = detailLine(text.replace(/^slash worker (?:exited|closed pipe:?|start failed:?)\s*/, ''))

    return [t('userMessages.rpc.slashCrashed', command), detail]
      .filter(Boolean)
      .join('\n')
  }

  return describeRpcError(err)
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

type TurnCopy = { hint: string; hintNoRetry?: string; title: string }
type TurnCopyTable = Record<string, TurnCopy | undefined>

// error_surface.code (snake_case wire values) → catalog leaf under userMessages.turn.code.
const TURN_CODE_KEY: Record<string, keyof Translations['userMessages']['turn']['code']> = {
  auth: 'auth',
  auth_permanent: 'authPermanent',
  billing: 'billing',
  billing_unverified: 'billingUnverified',
  content_policy_blocked: 'contentPolicyBlocked',
  context_overflow: 'contextOverflow',
  format_error: 'formatError',
  model_not_found: 'modelNotFound',
  overloaded: 'overloaded',
  payload_too_large: 'payloadTooLarge',
  provider_policy_blocked: 'providerPolicyBlocked',
  rate_limit: 'rateLimit',
  server_error: 'serverError',
  ssl_cert_verification: 'sslCertVerification',
  timeout: 'timeout',
  upstream_blocked: 'upstreamBlocked',
  upstream_rate_limit: 'upstreamRateLimit'
}

const turnCopyFor = (code: string, layer: string): TurnCopy => {
  const turn = messages().userMessages.turn
  const codeKey = TURN_CODE_KEY[code]
  const byCode = codeKey ? (turn.code as TurnCopyTable)[codeKey] : undefined

  return byCode ?? (turn.layer as TurnCopyTable)[layer] ?? turn.fallback
}

export interface TurnFailure {
  error?: null | string
  error_surface?: ErrorSurface | null | Record<string, unknown>
  recoverable?: boolean | null
}

/** Plain title + dimmed detail + next step for a failed turn with no reply text. */
export const describeTurnFailure = (payload: TurnFailure): string => {
  const surface = (payload.error_surface ?? {}) as { code?: unknown; layer?: unknown; provider?: unknown }
  const code = typeof surface.code === 'string' ? surface.code : ''
  const layer = typeof surface.layer === 'string' ? surface.layer : ''
  const copy = turnCopyFor(code, layer)

  const title =
    typeof surface.provider === 'string' && surface.provider
      ? t('userMessages.turn.withProvider', copy.title, surface.provider)
      : copy.title

  // The backend always sets recoverable=true on a turn error; error_surface.retryable
  // is the signal that actually says whether /retry can help.
  const retryable = (surface as { retryable?: unknown }).retryable !== false && payload.recoverable !== false
  const nextStep = retryable ? copy.hint : (copy.hintNoRetry ?? copy.hint)
  const raw = (payload.error ?? '').replace(/^Error:\s*/, '')

  return [t('userMessages.turn.notAnswered', title), detailLine(raw), nextStep].filter(Boolean).join('\n')
}

/** True when the assistant slot carries nothing but the backend's "Error: …" fallback text. */
export const isBareErrorText = (text: string, error: null | string | undefined): boolean => {
  const t = text.trim()

  return !t || t === `Error: ${error ?? ''}`.trim() || t === (error ?? '').trim()
}

// ── Withdrawn password / secret prompts ───────────────────────────────────

// server-request method → catalog leaf under userMessages.promptTimeout.
const PROMPT_TIMEOUT_KEY: Record<string, keyof Translations['userMessages']['promptTimeout']> = {
  secret: 'secret',
  sudo: 'sudo',
  'vault.code': 'vaultCode',
  'vault.save_login': 'vaultSaveLogin',
  'vault.unlock_prompt': 'vaultUnlockPrompt'
}

export const promptTimeoutNotice = (method: string | undefined, reason: string | undefined): null | string => {
  const key = reason === 'timeout' && method ? PROMPT_TIMEOUT_KEY[method] : undefined

  return key ? messages().userMessages.promptTimeout[key] : null
}

// ── session.info warnings ─────────────────────────────────────────────────

const MISSING_KEY_RE = /^No API key configured for provider '([^']*)'/

/** The backend's credential warning names the break; add the fix (/model saves a key in place). */
export const describeCredentialWarning = (warning: string): string => {
  const m = MISSING_KEY_RE.exec(warning)

  if (!m) {
    return warning
  }

  return t('userMessages.credential.missingKey', m[1] || t('userMessages.credential.currentProvider'))
}

// ── Empty states ──────────────────────────────────────────────────────────

export const noSkillsInstalled = (): string => t('userMessages.skills.noneInstalled')
