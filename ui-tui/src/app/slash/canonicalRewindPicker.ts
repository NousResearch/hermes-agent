// `/rewind` (and Esc Esc on an idle, empty composer): pick any earlier prompt, edit it, resend it.
// The cut is the same revision-fenced `session.mutate rewind` that `/undo` and `/retry` use, aimed at
// the picked row instead of the newest one — the owner's form of the legacy `prompt.submit
// truncate_before_row_id` that Desktop's edit-and-resend sends. Nothing is cut until the edit is sent.

import { introMsg, toTranscriptMessages } from '../../domain/messages.js'
import { t } from '../../i18n/runtime.js'
import { asRpcResult, rpcErrorMessage } from '../../lib/rpc.js'
import type { SessionInfo } from '../../types.js'
import type { RewindTarget } from '../interfaces.js'
import { patchOverlayState } from '../overlayStore.js'
import { captureDestination, type SubmissionDestination } from '../submissionDestination.js'
import { getUiState, patchUiState } from '../uiStore.js'

import { isUserTurn, retryText, rewindBusyText } from './canonicalSessionCommands.js'
import { mutateCanonical, type MutationSnapshot } from './canonicalSessionControls.js'
import type { SlashRunCtx } from './types.js'

/** The earlier prompts a rewind can return to, oldest first. A turn carrying media or annotated parts
 * has no lossless replay (same rule as /retry), so it keeps its number but is not offered. */
export function rewindTargets(snapshot: Pick<MutationSnapshot, 'messages' | 'stored_messages'>): RewindTarget[] {
  return (snapshot.stored_messages ?? snapshot.messages ?? []).filter(isUserTurn).flatMap((row, index) => {
    const text = retryText(row.content)

    return text ? [{ rowId: row.row_id as number, text, turn: index + 1 }] : []
  })
}

export function openRewindPicker(ctx: SlashRunCtx) {
  const sid = ctx.sid

  if (!sid) {
    return ctx.transcript.sys(t('slashCmd.core.rewind.nothing'))
  }

  if (!ctx.gateway.gw?.isCanonical) {
    return ctx.transcript.sys(t('slashCmd.core.rewind.unsupported'))
  }

  if (ctx.ui.busy) {
    return ctx.transcript.sys(t('slashCmd.core.busy.replying', 'rewind'))
  }

  void ctx.gateway.gw
    .request('session.resume', { session_id: sid })
    .then(raw => {
      if (ctx.stale()) {
        return
      }

      const targets = rewindTargets(asRpcResult<MutationSnapshot>(raw) ?? {})

      if (!targets.length) {
        return ctx.transcript.sys(t('slashCmd.core.rewind.nothing'))
      }

      const destination = captureDestination()

      patchOverlayState({
        rewind: { targets, onSubmit: (target, text) => void rewindAndSend(ctx, sid, target, text, destination) }
      })
    })
    .catch(ctx.guardedErr)
}

/** Cut the session back to just before `target`, repaint the transcript from the owner's snapshot, then
 * send the edited text as an ordinary prompt. A refusal (busy, stale revision) cuts nothing.
 *
 * The picker is closed by now, so the edit is never dropped: once the cut has committed it is owed to
 * `sid` as its next prompt, and any later failure (repaint fetch, session switch) leaves it in that
 * session's pending queue; if nothing was cut it goes back into the composer. */
export async function rewindAndSend(
  ctx: SlashRunCtx,
  sid: string,
  target: RewindTarget,
  text: string,
  destination: SubmissionDestination = captureDestination()
) {
  const stale = () => getUiState().sid !== sid
  const gw = ctx.gateway.gw
  let cut = false

  const keepEdit = () => {
    if (cut) {
      ctx.composer.enqueue(text, text, destination)
    } else if (!stale()) {
      ctx.composer.setInput(text)
    }
  }

  try {
    const mutation = await mutateCanonical(gw, sid, 'rewind', { target_message_id: target.rowId }, stale)
    const removed = Number(mutation?.result?.rewound_count ?? 0)

    cut = removed > 0

    if (!mutation || stale()) {
      return keepEdit()
    }

    if (!cut) {
      keepEdit()

      return ctx.transcript.sys(t('slashCmd.core.rewind.nothing'))
    }

    const snapshot = asRpcResult<{ info?: SessionInfo; messages?: unknown[] }>(
      await gw.request('session.resume', { session_id: sid })
    )

    if (stale()) {
      return keepEdit()
    }

    const info = { ...getUiState().info, ...snapshot?.info } as SessionInfo

    ctx.transcript.setHistoryItems([introMsg(info), ...toTranscriptMessages(snapshot?.messages ?? [])])
    patchUiState({ info })
    ctx.transcript.sys(t('slashCmd.core.rewind.done', String(target.turn), String(removed)))
    ctx.transcript.send(text)
  } catch (error) {
    keepEdit()

    if (!stale()) {
      ctx.transcript.sys(rewindBusyText(error, 'rewind') ?? `error: ${rpcErrorMessage(error)}`)
    }
  }
}
