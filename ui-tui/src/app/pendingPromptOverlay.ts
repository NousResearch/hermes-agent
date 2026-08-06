import type { SessionPendingPrompt } from '../gatewayTypes.js'
import type { OverlayState } from './interfaces.js'

/**
 * The overlay a blocking prompt maps to, and the status line that goes with it.
 *
 * Two callers need this mapping and must not drift apart: the live event
 * handler, when the prompt arrives as a `*.request` event, and session
 * rehydration, when the gateway replays a prompt that was already outstanding
 * on a session the user switched back to (#67265). Duplicating the mapping
 * meant a field added to a payload would silently be dropped on the
 * rehydration path only — visible as an overlay that renders with a missing
 * question or env var, but only after a session switch.
 *
 * This returns a description rather than applying it: the two callers push the
 * status differently (the event handler's `setStatus` also cancels a pending
 * thinking-status timer), and only the mapping is worth sharing.
 */
export interface PendingPromptOverlay {
  overlay: Partial<OverlayState>
  status: string
}

export const pendingPromptOverlay = (prompt?: null | SessionPendingPrompt): null | PendingPromptOverlay => {
  if (!prompt) {
    return null
  }

  switch (prompt.event) {
    case 'clarify.request':
      return {
        overlay: {
          clarify: {
            choices: prompt.payload.choices,
            question: prompt.payload.question,
            requestId: prompt.payload.request_id
          }
        },
        status: 'waiting for input…'
      }
    case 'sudo.request':
      return {
        overlay: { sudo: { requestId: prompt.payload.request_id } },
        status: 'sudo password needed'
      }
    case 'secret.request':
      return {
        overlay: {
          secret: {
            envVar: prompt.payload.env_var,
            prompt: prompt.payload.prompt,
            requestId: prompt.payload.request_id
          }
        },
        status: 'secret input needed'
      }
  }

  return null
}
