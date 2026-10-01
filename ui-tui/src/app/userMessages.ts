// User-facing wording for gateway/transport failures in the TUI. Pure functions
// so the copy — and the "what happened / what to do" shape — is unit-testable
// without rendering. Every slash command cited here exists in
// ui-tui/src/app/slash/commands (/logs, /retry, /model, /update, /resume,
// /sessions, /quit) and `hermes doctor` is a real subcommand.

import type { ErrorSurface } from '@hermes/shared/gateway-events'

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

  return `Details: ${text.length > DETAIL_LIMIT ? `${text.slice(0, DETAIL_LIMIT - 1)}…` : text}`
}

// ── Backend process lifecycle ─────────────────────────────────────────────

export const BACKEND_RESTARTING =
  'Hermes stopped unexpectedly — restarting and reopening your chat (the reply in progress was lost).'

export const BACKEND_RESTARTING_ACTIVITY = 'Hermes stopped unexpectedly · restarting…'

// Attached (dashboard / embedded) mode: only the socket dropped; Hermes and any
// reply in progress are still alive on the backend and come back on reconnect.
export const CONNECTION_LOST = 'Connection to Hermes lost — reconnecting and reopening your chat…'

export const CONNECTION_LOST_ACTIVITY = 'connection lost · reconnecting…'

export const backendGaveUp = (code: null | number, lastLine?: string): string => {
  const exit = code === null ? '' : ` (exit code ${code})`
  const detail = detailLine(lastLine)

  return [
    `Hermes stopped${exit} and could not be restarted. Your chat is saved.`,
    detail,
    'Hermes keeps trying to reconnect in the background and reopens this chat when it succeeds; if it does not, type /resume.',
    'Type /logs for the full log, or /quit and run `hermes doctor` to check the install.'
  ]
    .filter(Boolean)
    .join('\n')
}

export const BACKEND_GAVE_UP_ACTIVITY = 'Hermes stopped · /logs for details'

/** Last line of the backend log tail that is not our own [lifecycle]/[startup] bookkeeping. */
export const lastStderrLine = (tail: string): string | undefined =>
  tail
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !/^\[(?:lifecycle|startup|protocol|sidecar|spawn)\]/.test(l))
    .at(-1)

export const backendReconnecting = (attempt: number | undefined, delayMs: number | undefined): string => {
  const secs = Math.max(1, Math.round((delayMs ?? 1000) / 1000))
  const n = attempt && attempt > 0 ? ` (attempt ${attempt})` : ''

  return `retrying in ${secs}s${n}`
}

export const BACKEND_SLOW_START =
  'Hermes is taking longer than usual to start. Still waiting… If it never connects: /logs shows the last backend output; /quit and run `hermes doctor`.'

export const BACKEND_SLOW_START_STATUS = 'still starting…'

// ── stderr noise ──────────────────────────────────────────────────────────

// Only real failures: a traceback, a CRITICAL log line, an `XxxError:` / `XxxException:`
// head, or the gateway's own turn/exit markers. Dependency `DeprecationWarning` /
// `UserWarning` lines are noise and stay in /logs only.
const STDERR_PROBLEM_RE = /Traceback|\b[A-Z][A-Za-z]*(?:Error|Exception)\b:|CRITICAL|\[gateway-turn\]|\[gateway-exit\]/

/** Only lines that look like a failure earn an activity row; the rest stay in /logs. */
export const stderrLooksLikeProblem = (line: string): boolean => STDERR_PROBLEM_RE.test(line)

export const stderrProblemActivity = (line: string): string => {
  const m = /([A-Z][A-Za-z]*(?:Error|Exception)):/.exec(line)
  const what = m ? ` (${m[1]})` : ''

  return `Something went wrong inside Hermes${what} · /logs for details`
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

export const VERSION_SKEW_MESSAGE =
  'The terminal UI and the Hermes backend are out of sync (different versions). Run /update, or exit and run `hermes update`, then start the TUI again.'

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
    () =>
      'This chat is no longer attached to the backend (it was idle or the backend restarted). Your history is saved: type /resume to reopen it.'
  ],
  [
    (_code, text) => NOT_CONNECTED_RE.test(text),
    () =>
      'Hermes is not connected right now, so that was not sent. It reconnects automatically; wait a moment and try again, or type /logs if this persists.'
  ],
  [
    (_code, text) => TIMED_OUT_RE.exec(text),
    m =>
      `Hermes did not answer within ${m?.[1] ?? '?'}s. Try again; if it keeps happening, type /logs and report the last lines.`
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
  const text = message ?? (typeof err === 'string' && err.trim() ? err : 'request failed')

  if (isVersionSkewError(err)) {
    logReplacedWireText(code, text)

    return VERSION_SKEW_MESSAGE
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
    return `/${command} did not finish: the command helper timed out. Try again; if it keeps happening, type /logs and report the last lines.`
  }

  if (/slash worker (?:exited|closed pipe|start failed)/.test(text)) {
    const detail = detailLine(text.replace(/^slash worker (?:exited|closed pipe:?|start failed:?)\s*/, ''))

    return [`/${command} did not finish: the command helper crashed. Try again; type /logs for the trace.`, detail]
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

const TURN_CODE_COPY: Record<string, [string, string]> = {
  auth: ['The model provider rejected the API key', 'Fix the key with /model, then /retry.'],
  auth_permanent: ['The model provider rejected the API key', 'Fix the key with /model, then /retry.'],
  billing: ['The model provider reports no credit left', 'Top up the account or switch with /model.'],
  billing_unverified: ['The model provider reports no credit left', 'Top up the account or switch with /model.'],
  content_policy_blocked: ['The model provider refused this request (content policy)', 'Rephrase and send again.'],
  context_overflow: ['The conversation is too long for this model', 'Run /compress, then /retry.'],
  format_error: ['The model provider rejected the request format', 'Try /retry; if it persists, switch with /model.'],
  model_not_found: ['The model provider does not know this model', 'Pick another model with /model.'],
  overloaded: ['The model provider is overloaded', 'Wait a moment, then /retry.'],
  payload_too_large: ['The request was too large for this model', 'Run /compress, then /retry.'],
  provider_policy_blocked: ['The model provider refused this request (account policy)', 'Switch with /model.'],
  rate_limit: ['The model provider is rate-limiting requests', 'Wait a moment, then /retry.'],
  server_error: ['The model provider had an internal error', 'Wait a moment, then /retry.'],
  ssl_cert_verification: [
    'The connection to the model provider could not be verified (TLS)',
    "Check the endpoint's certificate, then /retry."
  ],
  timeout: ['The model provider did not answer in time', 'Try /retry; if it keeps happening, switch with /model.'],
  upstream_blocked: [
    'A firewall/CDN in front of the model provider blocked the request',
    "Set a User-Agent via the provider's extra_headers, or switch with /model."
  ],
  upstream_rate_limit: ['The model provider is rate-limiting requests', 'Wait a moment, then /retry.']
}

const TURN_LAYER_COPY: Record<string, [string, string]> = {
  auth: ['The model provider rejected the credentials', 'Fix them with /model, then /retry.'],
  billing: ['The model provider reports no credit left', 'Top up the account or switch with /model.'],
  disk: ['The disk is full, so Hermes could not save the turn', 'Free some space, then /retry.'],
  endpoint: ['Your custom model endpoint did not answer', 'Check the endpoint is running, then /retry.'],
  gateway: ['Hermes hit an internal error while running this turn', 'Send /retry; type /logs for the trace.'],
  provider: ['The model provider returned an error', 'Send /retry, or switch with /model.'],
  streaming: ['The connection to the model provider dropped mid-reply', 'Send /retry.']
}

const TURN_DEFAULT_COPY: [string, string] = ['The request failed', 'Send /retry, or switch with /model.']

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
  const provider = typeof surface.provider === 'string' && surface.provider ? ` (${surface.provider})` : ''
  const [title, hint] = TURN_CODE_COPY[code] ?? TURN_LAYER_COPY[layer] ?? TURN_DEFAULT_COPY
  // The backend always sets recoverable=true on a turn error; error_surface.retryable
  // is the signal that actually says whether /retry can help.
  const retryable = (surface as { retryable?: unknown }).retryable !== false && payload.recoverable !== false
  const nextStep = retryable ? hint : hint.replace(/(?:Send|Try) \/retry/, 'Pick another model with /model')
  const raw = (payload.error ?? '').replace(/^Error:\s*/, '')

  return [`${title}${provider}. Your message was not answered.`, detailLine(raw), nextStep].filter(Boolean).join('\n')
}

/** True when the assistant slot carries nothing but the backend's "Error: …" fallback text. */
export const isBareErrorText = (text: string, error: null | string | undefined): boolean => {
  const t = text.trim()

  return !t || t === `Error: ${error ?? ''}`.trim() || t === (error ?? '').trim()
}

// ── Withdrawn password / secret prompts ───────────────────────────────────

const PROMPT_TIMEOUT_COPY: Record<string, string> = {
  secret:
    'Secret prompt closed: no answer in time, so the step that needed it was skipped. Send your request again when you are ready to enter it.',
  sudo: 'Password prompt closed: no answer in time, so the command was skipped. Send your request again when you are ready to enter it.',
  'vault.code':
    'Verification-code prompt closed: no answer in time, so the sign-in was skipped. Send your request again when you have the code.',
  'vault.save_login':
    'Save-login prompt closed: no answer in time, so nothing was saved. Send your request again when you are ready.',
  'vault.unlock_prompt':
    'Unlock prompt closed: no answer in time, so the password manager stayed locked. Send your request again when you are ready to unlock it.'
}

export const promptTimeoutNotice = (method: string | undefined, reason: string | undefined): null | string =>
  reason === 'timeout' && method ? (PROMPT_TIMEOUT_COPY[method] ?? null) : null

// ── session.info warnings ─────────────────────────────────────────────────

const MISSING_KEY_RE = /^No API key configured for provider '([^']*)'/

/** The backend's credential warning names the break; add the fix (/model saves a key in place). */
export const describeCredentialWarning = (warning: string): string => {
  const m = MISSING_KEY_RE.exec(warning)

  if (!m) {
    return warning
  }

  const provider = m[1] || 'the current provider'

  return `No API key is set for ${provider}, so messages will fail. Type /model, pick ${provider}, and paste a key (or run /setup).`
}

// ── Empty states ──────────────────────────────────────────────────────────

export const NO_SKILLS_INSTALLED =
  'No skills installed yet. Type /skills browse to see the catalog, or /skills install <name>.'
