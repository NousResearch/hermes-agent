import type { ServerRequestMap } from '@hermes/shared/gateway-events'
import type { ServerRequest } from '@hermes/shared/json-rpc-channel'

import { patchOverlayState } from './overlayStore.js'
import { rememberServerRequest } from './serverRequestStore.js'

export interface ServerRequestHandlerContext {
  ringPromptBell: () => void
  setStatus: (status: string) => void
}

/** The subset of server→client methods the terminal can answer; every other
 *  generated method is absent on purpose (see the doc comment below). */
type ServerRequestHandlers = { [M in keyof ServerRequestMap]?: (request: ServerRequest<M>) => void }

/**
 * The Ink TUI's answer to the backend's server→client requests
 * (`tui_gateway/server_requests.py`). Each entry opens its overlay card;
 * the card's answer path resolves the request through `serverRequestStore`.
 * Methods the terminal cannot answer (desktop GUI bridges: `preview.*`,
 * `window.read`, `tour`, `terminal.read`, the other vault
 * prompts) have no entry, so the caller answers `-32601` and the tool fails
 * fast instead of waiting out its deadline.
 */
export function createServerRequestHandler(
  ctx: ServerRequestHandlerContext
): <M extends keyof ServerRequestMap>(request: ServerRequest<M>) => boolean {
  const { ringPromptBell, setStatus } = ctx

  const open = <M extends keyof ServerRequestMap>(request: ServerRequest<M>, status: string) => {
    rememberServerRequest(request)
    setStatus(status)

    if (!request.replayed) {
      ringPromptBell()
    }
  }

  const handlers: ServerRequestHandlers = {
    approval: request => {
      const p = request.params

      patchOverlayState({
        approval: {
          // Only an explicit false (tirith warning) drops the permanent-allow option.
          allowPermanent: p.allow_permanent !== false,
          choices: p.choices.length ? p.choices : undefined,
          command: p.command,
          description: p.description || 'dangerous command',
          requestId: request.id,
          smartDenied: p.smart_denied === true
        }
      })
      open(request, 'approval needed')
    },

    clarify: request => {
      patchOverlayState({
        clarify: {
          answers: (request.params.kind === 'batch' ? request.params.answers : null) ?? {},
          params: request.params,
          requestId: request.id
        }
      })
      open(request, 'waiting for input…')
    },

    secret: request => {
      patchOverlayState({
        secret: { envVar: request.params.env_var, prompt: request.params.prompt, requestId: request.id }
      })
      open(request, 'secret input needed')
    },

    sudo: request => {
      patchOverlayState({ sudo: { requestId: request.id } })
      open(request, 'sudo password needed')
    },

    'vault.unlock_prompt': request => {
      const p = request.params

      patchOverlayState({
        vaultUnlock: { backend: p.backend, displayName: p.display_name, requestId: request.id }
      })
      open(request, `unlock ${p.display_name}`)
    }
  }

  return request => {
    const handler = handlers[request.method]

    if (!handler) {
      return false
    }

    handler(request)

    return true
  }
}
