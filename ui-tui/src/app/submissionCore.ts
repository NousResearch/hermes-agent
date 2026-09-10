import type { GatewayClient } from '../gatewayClient.js'
import type { InputDetectDropResponse, PromptSubmitResponse, SessionActivateResponse } from '../gatewayTypes.js'
import type { QueueItem } from '../hooks/useQueue.js'
import { stageImagePath } from '../lib/imageAttachments.js'
import { pendingInputOwner, savePendingInput } from '../lib/pendingInputs.js'
import type { Msg } from '../types.js'

import { markBubbleShown } from './pendingBubbles.js'
import { captureDestination, isCurrentDestination, type SubmissionDestination } from './submissionDestination.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

const SESSION_BUSY_RE = /session busy|waiting for model response/i

export const isSessionBusyError = (e: unknown) => e instanceof Error && SESSION_BUSY_RE.test(e.message)

export interface SubmitPromptDeps {
  appendMessage: (msg: Msg) => void
  enqueue: (text: string, display?: string, destination?: SubmissionDestination) => void
  expand: (text: string) => string
  gw: GatewayClient
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
//
// `displayOverride` is what the transcript shows when it differs from what the
// agent receives — a `/skill` invocation expands into the whole skill body, and
// that scaffolding is model-facing only.
export function submitPrompt(
  text: string,
  deps: SubmitPromptDeps,
  showUserMessage = true,
  displayOverride?: string,
  opts: { attachments?: Array<{ path: string; mime: string }>; skipDetectDrop?: boolean; destination?: SubmissionDestination; queueItem?: QueueItem; behindTurn?: boolean } = {}
): void {
  const destination = opts.destination ?? captureDestination()
  const owner = pendingInputOwner(opts.queueItem?.ownerDestination ?? destination)
  const { sid } = owner
  const focused = () => isCurrentDestination(owner)
  // A busy-time admission joins the authority FIFO behind the running turn:
  // it must not touch the in-flight stream buffer, status or transcript. Its
  // user bubble is painted when the fanout reports the row started.
  const ownsTurn = () => focused() && !opts.behindTurn

  if (!sid) {
    return deps.sys('session not ready yet')
  }

  // An identityless write cannot be deduplicated, even after the owner upgrades.
  if (opts.queueItem?.legacyAttempted) {
    opts.queueItem.settle?.(false)

    if (focused()) {
      deps.sys('legacy delivery unconfirmed — check the session before sending a new input; retained input was not resent')
      patchUiState({ status: 'delivery unconfirmed' })
    }

    return
  }

  // Close the async-busy gap up front, before the detect_drop round-trip.
  if (ownsTurn()) {
    markSubmitting()
  }

  const startSubmit = (displayText: string, submitText: string, show = true) => {
    if (ownsTurn()) {
      turnController.clearStatusTimer()
      deps.setLastUserMsg(text)

      if (show) {
        deps.appendMessage({ role: 'user', text: displayOverride || displayText })
      }

      patchUiState({ busy: true, status: 'running…' })
      turnController.bufRef = ''
      turnController.interrupted = false
    }

    const item = opts.queueItem

    if (item) {
      item.preparedText ??= submitText
      item.attachments ??= opts.attachments?.map(attachment => ({ ...attachment }))

      if (opts.behindTurn) { item.queued = true }

      if (ownsTurn() && show) { markBubbleShown(item.submissionId) }
      savePendingInput(item)
    }

    deps.gw
      .request<PromptSubmitResponse>(item?.controlMethod ?? 'prompt.submit', {
        session_id: sid,
        text: item?.preparedText ?? submitText,
        ...((item?.attachments ?? opts.attachments)?.length ? { attachments: item?.attachments ?? opts.attachments } : {}),
        ...(item?.controlMethod ? { execution_generation: item.executionGeneration } : {}),
        ...(item ? { submission_id: item.submissionId, ...(item.controlMethod ? {} : { queued: item.queued !== false }) } : {})
      })
      .then(r => {
        if (item) {
          const accepted =
            (r?.input_id ?? r?.admission_id) === item.submissionId &&
            Boolean(destination.storedSid) &&
            r?.target_session_id === destination.storedSid &&
            r?.target_profile_home === destination.profileHome &&
            ['queued', 'started', 'terminal', 'unknown'].includes(r?.status ?? '')

          item.settle?.(accepted)

          // A replay receipt ends this admission, not necessarily the current
          // execution. Refresh the same live attachment without its transcript.
          if (accepted && focused() && ['terminal', 'unknown'].includes(r.status ?? '')) {
            const observed = getUiState().info

            void deps.gw.request<SessionActivateResponse>('session.activate', { session_id: sid, omit_messages: true })
              .then(snapshot => {
                const current = getUiState().info
                const info = snapshot.info

                // Push lifecycle events (including same-generation completion)
                // win over an in-flight snapshot; attachment changes win too.
                if (!focused() || current !== observed || snapshot.session_id !== sid ||
                    (snapshot.session_key || info?.stored_session_id) !== owner.storedSid ||
                    typeof snapshot.running !== 'boolean' ||
                    (current?.execution_generation !== undefined &&
                      (info?.execution_epoch !== current.execution_epoch ||
                        !Number.isSafeInteger(info?.execution_generation) ||
                        (info?.execution_generation ?? -1) < current.execution_generation))) {
                  return
                }

                patchUiState({ busy: snapshot.running, status: snapshot.running ? 'running…' : 'ready',
                  info: current && { ...current, execution_epoch: info?.execution_epoch ?? current.execution_epoch,
                    execution_generation: info?.execution_generation ?? current.execution_generation,
                    running: snapshot.running } })
              })
              .catch((error: Error) => {
                if (focused()) { deps.sys(`execution status unconfirmed: ${error.message}`) }
              })
          }

          if (!accepted && focused()) {
            deps.sys('admission not confirmed — input retained; use Alt+K to retry with the same identity')
            patchUiState({ status: 'admission unconfirmed' })
          }
        }

        // The gateway consumed a typed voice stop phrase server-side (voice
        // chat ended, no turn started) — release the busy latch; the
        // voice.transcript {stop_phrase} event handles the mode flags + notice.
        if (r?.voice_stopped && focused()) {
          patchUiState({ busy: false, status: 'ready' })
        }
      })
      .catch(async (e: Error & { code?: number }) => {
        // 4094 is a pre-admission refusal, not an ambiguous write. Special
        // compute modes still support legacy submit; retry only this refusal,
        // keeping the prepared payload, destination and queue mode unchanged.
        if (item && e.code === 4094 && !deps.gw.isCanonical && !item.attachments?.length) {
          if (focused()) {deps.sys('durable admission unavailable for this session — using legacy delivery')}

          try {
            item.legacyAttempted = true
            savePendingInput(item)

            const r = await deps.gw.request<PromptSubmitResponse>('prompt.submit', {
              session_id: sid,
              text: item.preparedText ?? submitText,
              queued: item.queued !== false
            })

            const accepted = Boolean(r?.voice_stopped || ['streaming', 'queued', 'steered', 'redirected'].includes(r?.status ?? ''))
            item.settle?.(accepted)

            if (focused()) {
              if (r?.voice_stopped) {patchUiState({ busy: false, status: 'ready' })}
              else if (!accepted) {
                deps.sys('legacy delivery unconfirmed — input retained; check the session before sending a new input')
                patchUiState({ status: 'delivery unconfirmed' })
              }
            }
          } catch (error) {
            item.settle?.(false)

            if (focused()) {
              deps.sys(`legacy delivery unconfirmed: ${error instanceof Error ? error.message : String(error)} — input retained; check the session before sending a new input`)
              patchUiState({ status: 'delivery unconfirmed' })
            }
          }

          return
        }

        if (item) {
          item.settle?.(false)

          if (focused()) {
            deps.sys(`input retained: ${e.message} — Alt+K retries the same submission`)
            patchUiState({ status: 'admission unconfirmed' })
          }

          return
        }

        // Defensive: prompt.submit no longer rejects a mid-turn send with
        // "session busy" (the gateway queues it and returns success), but keep
        // the re-queue path as a safety net for any future/legacy gateway that
        // still errors, so a message is never silently dropped.
        if (isSessionBusyError(e)) {
          deps.enqueue(submitText, displayOverride, destination)

          if (!focused()) {
            return
          }

          patchUiState({ busy: true, status: 'queued for next turn' })

          return deps.sys(`queued: "${submitText.slice(0, 50)}${submitText.length > 50 ? '…' : ''}"`)
        }

        if (!focused()) {
          return
        }

        deps.sys(`error: ${e.message}`)
        patchUiState({ busy: false, status: 'ready' })
      })
  }

  // Always ask the backend whether this looks like a file drop. The backend's
  // _detect_file_drop handles paths with spaces, quotes, Windows drive letters,
  // and escaped characters correctly. Literal submissions (startup -q queries)
  // skip it: launcher-provided text must reach the agent untouched.
  //
  // No notice is emitted for a match: an image dropped into the composer already
  // shows as an `[[ Image N ]]` token, and a matched non-image path is rewritten
  // in place. Announcing it a second time above the status bar was the old
  // out-of-band attachment UI.
  if (opts.queueItem?.preparedText !== undefined) {
    return startSubmit(text, opts.queueItem.preparedText, showUserMessage)
  }

  if (opts.skipDetectDrop) {
    return startSubmit(text, deps.expand(text), showUserMessage)
  }

  if (deps.gw.isCanonical) {
    if (/^(?:["']?(?:[/.~]|[A-Za-z]:[/\\])|file:\/\/)/.test(text) && /\.(?:png|jpe?g|gif|webp)(?:["']?)(?:\s|$)/i.test(text)) {
      void stageImagePath(text, deps.gw, destination).then(image => {
        opts.attachments = [{ path: image.path, mime: image.mime }]
        startSubmit(text, image.remainder || 'What do you see in this image?', showUserMessage)
      }).catch((error: Error) => {
        opts.queueItem?.settle?.(false)

        if (focused()) {
          deps.sys(`image not submitted: ${error.message} — input retained`)

          if (ownsTurn()) { patchUiState({ busy: false, status: 'image not submitted' }) }
        }
      })

      return
    }

    return startSubmit(text, deps.expand(text), showUserMessage)
  }

  deps.gw
    .request<InputDetectDropResponse>('input.detect_drop', { session_id: sid, text })
    .then(r => {
      if (!r?.matched) {
        return startSubmit(text, deps.expand(text), showUserMessage)
      }

      startSubmit(r.text || text, deps.expand(r.text || text), showUserMessage)
    })
    .catch(() => startSubmit(text, deps.expand(text), showUserMessage))
}
