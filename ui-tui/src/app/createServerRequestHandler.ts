import type { ServerRequest } from '@hermes/shared/json-rpc-channel'

import type { AttentionEvent } from '../lib/notify.js'
import type { ClarifyBatchQuestion } from '../types.js'

import { patchOverlayState } from './overlayStore.js'
import { rememberServerRequest } from './serverRequestStore.js'

export interface ServerRequestHandlerContext {
  /** Attention cue for a blocking prompt that just opened and waits on the
   *  user: bell (display.bell_on_prompt / notify_on_interact) plus the
   *  configured attention hook, if any. */
  notifyPromptAttention: (payload: PromptAttention) => void
  setStatus: (status: string) => void
}

export type PromptAttention = {
  event: AttentionEvent
  message: string
  session_id: null | string
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '')

const strList = (v: unknown): null | string[] =>
  Array.isArray(v) && v.length > 0 ? v.filter((c): c is string => typeof c === 'string') : null

/**
 * The Ink TUI's answer to the backend's server→client requests
 * (`tui_gateway/server_requests.py`). Each method opens its overlay card;
 * the card's answer path resolves the request through `serverRequestStore`.
 * Methods the terminal cannot answer (desktop GUI bridges: `preview.*`,
 * `window.read`, `tour`, `mcp.setup`, `terminal.read`, the vault card
 * prompts) return `false` so the channel answers `-32601` and the tool
 * fails fast instead of waiting out its deadline.
 */
export function createServerRequestHandler(ctx: ServerRequestHandlerContext): (request: ServerRequest) => boolean {
  const { notifyPromptAttention, setStatus } = ctx

  const open = (request: ServerRequest, status: string, attention: PromptAttention) => {
    rememberServerRequest(request)
    setStatus(status)

    if (!request.replayed) {
      notifyPromptAttention(attention)
    }
  }

  return request => {
    const p = request.params

    switch (request.method) {
      case 'clarify': {
        const batch: ClarifyBatchQuestion[] = (Array.isArray(p.questions) ? (p.questions as unknown[]) : [])
          .map(raw => (raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}))
          .filter(q => str(q.qid) && str(q.question).trim())
          .map(q => ({
            choices: strList(q.choices),
            multiSelect: q.multi_select === true,
            qid: str(q.qid),
            question: str(q.question).trim()
          }))

        const answers =
          p.answers && typeof p.answers === 'object'
            ? Object.fromEntries(
                Object.entries(p.answers as Record<string, unknown>).filter(
                  (entry): entry is [string, string] => typeof entry[1] === 'string'
                )
              )
            : {}

        patchOverlayState({
          clarify: batch.length
            ? { answers, choices: null, question: '', questions: batch, requestId: request.id }
            : { choices: strList(p.choices), question: str(p.question), requestId: request.id }
        })

        const questions = batch.length
          ? batch.map(q => q.question).join(' / ')
          : str(p.question).trim()

        open(
          request,
          'waiting for input…',
          { event: 'input.needed', message: questions, session_id: str(p.session_id) || null }
        )

        return true
      }

      case 'approval': {
        patchOverlayState({
          approval: {
            // Only an explicit false (tirith warning) drops the permanent-allow option.
            allowPermanent: p.allow_permanent !== false,
            choices: strList(p.choices) ?? undefined,
            command: str(p.command),
            description: str(p.description) || 'dangerous command',
            requestId: request.id,
            smartDenied: p.smart_denied === true
          }
        })
        open(request, 'approval needed', {
          event: 'approval.needed',
          message: `${str(p.description) || 'dangerous command'}: ${str(p.command)}`.trim(),
          session_id: str(p.session_id) || null
        })

        return true
      }

      case 'sudo':
        patchOverlayState({ sudo: { requestId: request.id } })
        open(request, 'sudo password needed', {
          event: 'sudo.needed',
          message: 'sudo password required',
          session_id: str(p.session_id) || null
        })

        return true

      case 'secret':
        patchOverlayState({ secret: { envVar: str(p.env_var), prompt: str(p.prompt), requestId: request.id } })
        open(request, 'secret input needed', {
          event: 'input.needed',
          message: str(p.prompt) || 'secret input required',
          session_id: str(p.session_id) || null
        })

        return true

      case 'vault.unlock_prompt':
        patchOverlayState({
          vaultUnlock: { backend: str(p.backend), displayName: str(p.display_name), requestId: request.id }
        })
        open(request, `unlock ${str(p.display_name)}`, {
          event: 'input.needed',
          message: `unlock ${str(p.display_name)}`.trim(),
          session_id: str(p.session_id) || null
        })

        return true

      default:
        return false
    }
  }
}
