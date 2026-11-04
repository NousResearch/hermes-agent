import type { ServerRequest } from '@hermes/shared/json-rpc-channel'

import { t } from '../i18n/runtime.js'
import type { ClarifyQuestion } from '../types.js'

import { patchOverlayState } from './overlayStore.js'
import { rememberServerRequest } from './serverRequestStore.js'
import { getUiState } from './uiStore.js'

export interface ServerRequestHandlerContext {
  /** The session this view shows (default: the UI store's); a live request for another session waits for its activation. */
  focusedSessionId?: () => null | string
  ringPromptBell: () => void
  setStatus: (status: string) => void
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '')

// The cards this handler opens; anything else is declined below and must keep failing fast.
const PROMPT_METHODS = new Set([
  'approval',
  'clarify',
  'secret',
  'sudo',
  'vault.code',
  'vault.save_login',
  'vault.unlock_prompt'
])

const strList = (v: unknown): null | string[] =>
  Array.isArray(v) && v.length > 0 ? v.filter((c): c is string => typeof c === 'string') : null

/**
 * The Ink TUI's answer to the backend's server→client requests
 * (`tui_gateway/server_requests.py`). Each method opens its overlay card;
 * the card's answer path resolves the request through `serverRequestStore`.
 * Methods the terminal cannot answer (desktop GUI bridges: `preview.*`,
 * `window.read`, `tour`, `mcp.setup`, `terminal.read`) return `false` so the
 * channel answers `-32601` and the tool fails fast instead of waiting out its
 * deadline.
 */
export function createServerRequestHandler(ctx: ServerRequestHandlerContext): (request: ServerRequest) => boolean {
  const { focusedSessionId = () => getUiState().sid, ringPromptBell, setStatus } = ctx

  const open = (request: ServerRequest, status: string) => {
    rememberServerRequest(request)
    setStatus(status)

    if (!request.replayed) {
      ringPromptBell()
    }
  }

  // A session kept running in the background (/new mid-turn, the switcher's `+ new`) still asks its
  // questions on this socket. Its card must not open over another session's view: the request stays
  // open on the backend unanswered, and activating that session hands it back as `open_requests`
  // (replayed), which opens it there. The status poll tells the user the session is waiting.
  const belongsElsewhere = (request: ServerRequest) => {
    const owner = request.params.session_id
    const focused = focusedSessionId()

    return (
      PROMPT_METHODS.has(request.method) &&
      !request.replayed &&
      typeof owner === 'string' &&
      Boolean(focused) &&
      owner !== focused
    )
  }

  return request => {
    const p = request.params

    if (belongsElsewhere(request)) {
      return true
    }

    switch (request.method) {
      case 'clarify': {
        const questions: ClarifyQuestion[] = (Array.isArray(p.questions) ? (p.questions as unknown[]) : [])
          .map(raw => (raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}))
          .filter(q => str(q.qid) && str(q.question).trim())
          .map(q => ({
            choices: strList(q.choices),
            multiSelect: q.multi_select === true,
            qid: str(q.qid),
            question: str(q.question).trim()
          }))

        if (!questions.length) {
          request.respond({})

          return true
        }

        const answers =
          p.answers && typeof p.answers === 'object'
            ? Object.fromEntries(
                Object.entries(p.answers as Record<string, unknown>)
                  .map(([qid, answer]): [string, unknown] => [qid, answer === null ? '' : answer])
                  .filter((entry): entry is [string, string] => typeof entry[1] === 'string')
              )
            : {}

        patchOverlayState({ clarify: { answers, questions, requestId: request.id } })
        open(request, t('session.status.waitingForInput'))

        return true
      }

      case 'approval': {
        patchOverlayState({
          approval: {
            // Only an explicit false drops the permanent-allow option.
            allowPermanent: p.allow_permanent !== false,
            choices: strList(p.choices) ?? undefined,
            command: str(p.command),
            description: str(p.description) || t('session.request.dangerousCommand'),
            requestId: request.id,
            smartDenied: p.smart_denied === true
          }
        })
        open(request, t('session.status.approvalNeeded'))

        return true
      }

      case 'sudo':
        patchOverlayState({ sudo: { requestId: request.id } })
        open(request, t('session.status.sudoPasswordNeeded'))

        return true

      case 'secret':
        patchOverlayState({ secret: { envVar: str(p.env_var), prompt: str(p.prompt), requestId: request.id } })
        open(request, t('session.status.secretInputNeeded'))

        return true

      case 'vault.unlock_prompt':
        patchOverlayState({
          vaultUnlock: { backend: str(p.backend), displayName: str(p.display_name), requestId: request.id }
        })
        open(request, t('session.status.unlockVault', str(p.display_name)))

        return true
      case 'vault.save_login': {
        const site = str(p.site) || str(p.origin)

        patchOverlayState({ vaultSaveLogin: { origin: str(p.origin), requestId: request.id, site } })
        open(request, t('session.status.saveLogin', site))

        return true
      }

      case 'vault.code': {
        const site = str(p.site)

        patchOverlayState({ vaultCode: { hint: str(p.hint), requestId: request.id, site } })
        open(request, t('session.status.verificationCode', site))

        return true
      }

      default:
        return false
    }
  }
}
