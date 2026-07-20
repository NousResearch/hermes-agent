import { attachedImageNotice } from '../domain/messages.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { InputDetectDropResponse, PromptSubmitResponse } from '../gatewayTypes.js'
import type { Msg } from '../types.js'

import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

const SESSION_BUSY_RE = /session busy|waiting for model response/i

export const isSessionBusyError = (e: unknown) => e instanceof Error && SESSION_BUSY_RE.test(e.message)

export interface SubmitPromptDeps {
  appendMessage: (msg: Msg) => void
  enqueue: (text: string) => void
  expand: (text: string) => string
  gw: GatewayClient
  onPromptAccepted?: (sid: string, eventSeq: number, generation: number) => void
  setLastUserMsg: (value: string) => void
  sys: (text: string) => void
}

// Optimistically flip the session to busy the INSTANT a prompt is accepted for
// submission — synchronously, before we await anything.
//
// This is the fix for the queue-mode race (display.busy_input_mode: queue):
// the submit path first fires an async `input.detect_drop` RPC and only marked
// the session busy inside that RPC's `.then`. A second Enter pressed inside
// that round-trip window read `busy === false` in dispatchSubmission and raced
// a second `prompt.submit` onto the backend instead of landing in the local
// queue. That produced the reported symptom: the second message "waited for
// the first to respond, then went to the queue", and the client lost track of
// it (the backend accepts a mid-turn submit as {status:"queued"} — a success,
// not an error — so the local drain effect that watches the client-side queue
// never fires, leaving the UI stuck on "analyzing…" until Ctrl+C).
//
// Marking busy at the choke point closes the gap for every caller: the mainline
// submit, queue-edit picks, and the drain effect all funnel through here.
export function markSubmitting(): void {
  patchUiState({ busy: true, status: 'running…' })
}

// Submit a ready prompt (already resolved to be neither a slash command nor a
// shell escape, with a live session). Pulled out of useSubmission so the
// synchronous-busy invariant above is unit-testable without React test infra.
export function submitPrompt(text: string, deps: SubmitPromptDeps, showUserMessage = true): void {
  const sid = getUiState().sid

  if (!sid) {
    return deps.sys('session not ready yet')
  }

  // Give this submit its own lifecycle identity and close the async-busy gap
  // up front, before the detect_drop round-trip.
  const generation = turnController.resetForSubmit()
  const ownsSubmission = () => getUiState().sid === sid && turnController.ownsPromptSubmission(generation)
  markSubmitting()

  const startSubmit = (displayText: string, submitText: string, show = true) => {
    if (!ownsSubmission()) {
      return
    }

    turnController.clearStatusTimer()
    deps.setLastUserMsg(text)

    if (show) {
      deps.appendMessage({ role: 'user', text: displayText })
    }

    patchUiState({ busy: true, status: 'running…' })
    const eventSeq = turnController.eventSeq

    deps.gw
      .request<PromptSubmitResponse>('prompt.submit', { session_id: sid, text: submitText })
      .then(r => {
        if (r?.status === 'streaming' && ownsSubmission()) {
          deps.onPromptAccepted?.(sid, eventSeq, generation)
        }
      })
      .catch((e: Error) => {
        if (!ownsSubmission() || !turnController.rejectPromptSubmission(generation)) {
          return
        }

        // Defensive: prompt.submit no longer rejects a mid-turn send with
        // "session busy" (the gateway queues it and returns success), but keep
        // the re-queue path as a safety net for any future/legacy gateway that
        // still errors, so a message is never silently dropped.
        if (isSessionBusyError(e)) {
          deps.enqueue(submitText)
          patchUiState({ busy: true, status: 'queued for next turn' })

          return deps.sys(`queued: "${submitText.slice(0, 50)}${submitText.length > 50 ? '…' : ''}"`)
        }

        deps.sys(`error: ${e.message}`)
        patchUiState({ busy: false, status: 'ready' })
      })
  }

  // Always ask the backend whether this looks like a file drop. The backend's
  // _detect_file_drop handles paths with spaces, quotes, Windows drive letters,
  // and escaped characters correctly.
  deps.gw
    .request<InputDetectDropResponse>('input.detect_drop', { session_id: sid, text })
    .then(r => {
      if (!ownsSubmission()) {
        return
      }

      if (!r?.matched) {
        return startSubmit(text, deps.expand(text), showUserMessage)
      }

      if (r.is_image) {
        turnController.pushActivity(attachedImageNotice(r))
      } else {
        turnController.pushActivity(`detected file: ${r.name}`)
      }

      startSubmit(r.text || text, deps.expand(r.text || text), showUserMessage)
    })
    .catch(() => {
      if (ownsSubmission()) {
        startSubmit(text, deps.expand(text), showUserMessage)
      }
    })
}
