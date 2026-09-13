import type {
  SessionActivateResponse,
  SessionResumeResponse
} from '../gatewayTypes.js'

import { patchOverlayState } from './overlayStore.js'

/**
 * Re-open the overlay for a prompt that is still blocking a resumed session.
 *
 * The gateway replays `pending_clarify` / `pending_sudo` / `pending_secret` on
 * `session.resume` and `session.activate` because the agent thread is parked on
 * the prompt's Event: a client that reattached after the `*.request` event was
 * emitted has nothing to render, so the session sits at `status: "waiting"`
 * with no answerable surface until the prompt times out.
 *
 * The shapes below deliberately mirror the live `*.request` handlers in
 * `createGatewayEventHandler.ts` — same fields, same `requestId` — so a
 * restored overlay and a freshly-emitted one are indistinguishable, and the
 * existing `*.respond` paths resolve either one.
 *
 * `pending_approval` is intentionally not handled here: approvals never enter
 * `_block()`'s registry, and the Ink approval overlay is driven by its own
 * `approval.request` flow.
 */
export const restorePendingPrompt = (
  r: SessionActivateResponse | SessionResumeResponse
): void => {
  if (r.pending_clarify) {
    patchOverlayState({
      clarify: {
        choices: r.pending_clarify.choices ?? null,
        question: r.pending_clarify.question ?? '',
        requestId: r.pending_clarify.request_id
      }
    })
    return
  }

  if (r.pending_sudo) {
    patchOverlayState({ sudo: { requestId: r.pending_sudo.request_id } })
    return
  }

  if (r.pending_secret) {
    patchOverlayState({
      secret: {
        envVar: r.pending_secret.env_var ?? '',
        prompt: r.pending_secret.prompt ?? '',
        requestId: r.pending_secret.request_id
      }
    })
  }
}
