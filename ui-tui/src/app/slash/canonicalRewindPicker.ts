// `/rewind` (and Esc, Esc on an idle, empty composer): Codex-style transcript browsing. Pick any
// earlier prompt, Enter puts it in the composer, and sending the edit rewinds to just before it.
// The cut is the same revision-fenced `session.mutate rewind` that `/undo` and `/retry` use, aimed at
// the picked row instead of the newest one — the owner's form of the legacy `prompt.submit
// truncate_before_row_id` that Desktop's edit-and-resend sends. Nothing is cut until the edit is sent.

import { accessSync, constants } from 'node:fs'

import { introMsg, toTranscriptMessages } from '../../domain/messages.js'
import { t } from '../../i18n/runtime.js'
import { splitImageNotes } from '../../lib/imageNotes.js'
import { asRpcResult, rpcErrorMessage } from '../../lib/rpc.js'
import { armPendingRewind, cancelBacktrackPrime, cancelPendingRewind, type RewindSubmission } from '../backtrackStore.js'
import type { RewindTarget } from '../interfaces.js'
import { patchOverlayState } from '../overlayStore.js'
import { captureDestination, type SubmissionDestination } from '../submissionDestination.js'
import { getUiState, patchUiState } from '../uiStore.js'

import { isUserTurn, retryText, rewindBusyText } from './canonicalSessionCommands.js'
import { applyBranchResult, mutateCanonical, type MutationSnapshot } from './canonicalSessionControls.js'
import type { SlashRunCtx } from './types.js'

const storedRowId = (row: Record<string, unknown>) =>
  Number.isSafeInteger(row.row_id) && (row.row_id as number) > 0 && !row.archived ? (row.row_id as number) : undefined

/** The earlier prompts a rewind can return to, oldest first. A turn carrying media or annotated parts
 * has no lossless replay (same rule as /retry), so it keeps its number but is not offered.
 * `throughRowId` is the last stored row before the prompt: a fork branches through it, so the child
 * holds everything before the prompt (the authority pulls in tool results that answer that row). */
export function rewindTargets(snapshot: Pick<MutationSnapshot, 'messages' | 'stored_messages'>): RewindTarget[] {
  const targets: RewindTarget[] = []
  let through: number | undefined
  let turn = 0

  for (const row of snapshot.stored_messages ?? snapshot.messages ?? []) {
    if (isUserTurn(row)) {
      turn += 1
      const text = retryText(row.content)

      if (text) {
        targets.push({ rowId: row.row_id as number, text, turn, ...(through ? { throughRowId: through } : {}) })
      }
    }

    through = storedRowId(row) ?? through
  }

  return targets
}

const readable = (path: string) => {
  try {
    accessSync(path, constants.R_OK)

    return true
  } catch {
    return false
  }
}

/** Put a picked prompt back in the composer as the user typed it: the image notes the model got are
 *  dropped and each image they name is attached again as a composer image. The paths are the owner's
 *  files and re-attaching reads them here, so a gateway on another machine (HERMES_TUI_GATEWAY_URL)
 *  whose files this machine cannot read keeps the notes in the text instead, and says so. */
function prefillPrompt(ctx: SlashRunCtx, stored: string) {
  const { images, text } = splitImageNotes(stored)

  if (images.length && process.env.HERMES_TUI_GATEWAY_URL?.trim() && !images.every(readable)) {
    ctx.composer.setInput(stored)

    return ctx.transcript.sys(t('slashCmd.core.rewind.imagesKeptAsNotes', images.length))
  }

  ctx.composer.setInput(text)

  if (images.length) {
    // The user may keep typing while the images stage: the restore follows the draft.
    void ctx.composer.attachImagePaths(images, { followTyping: true })
  }
}

export function openRewindPicker(ctx: SlashRunCtx) {
  const sid = ctx.sid

  cancelBacktrackPrime()
  // Browsing again starts over: an earlier armed edit no longer applies.
  cancelPendingRewind()

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

      const snapshot: Partial<MutationSnapshot> = asRpcResult<MutationSnapshot>(raw) ?? {}
      const targets = rewindTargets(snapshot)

      if (!targets.length) {
        return ctx.transcript.sys(t('slashCmd.core.rewind.noPrevious'))
      }

      const destination = captureDestination()

      const send = (target: RewindTarget, edit: RewindSubmission | string) =>
        void rewindAndSend(ctx, sid, target, edit, destination)

      patchOverlayState({
        rewind: {
          messages: toTranscriptMessages(snapshot.messages ?? []),
          targets,
          onEdit: target => {
            prefillPrompt(ctx, target.text)
            armPendingRewind({ send: edit => send(target, edit), sid, target })
          },
          onFork: target => void forkFrom(ctx, sid, target),
          onSubmit: send,
          onType: text => ctx.composer.setInput(prev => prev + text),
          onPaste: text => void ctx.composer.appendPaste(text)
        }
      })
    })
    .catch(ctx.guardedErr)
}

/** Cut the session back to just before `target`, repaint the transcript from the owner's snapshot, then
 * send the edited text as an ordinary prompt. A refusal (busy, stale revision) cuts nothing.
 *
 * Browsing is closed by now, so the edit is never dropped: once the cut has committed it is owed to
 * `sid` as its next prompt, and any later failure (repaint fetch, session switch) leaves it in that
 * session's pending queue; if nothing was cut it goes back into the composer. */
export async function rewindAndSend(
  ctx: SlashRunCtx,
  sid: string,
  target: RewindTarget,
  edit: RewindSubmission | string,
  destination: SubmissionDestination = captureDestination()
) {
  const { attachments, display, text } = typeof edit === 'string' ? { display: edit, text: edit } : edit
  const stale = () => getUiState().sid !== sid
  const gw = ctx.gateway.gw
  let cut = false

  // `text` is already expanded (paste chips resolved): the composer's tokens are gone by now.
  const keepEdit = () => {
    if (cut) {
      if (attachments?.length) {
        ctx.composer.enqueue(text, display, destination, attachments)
      } else {
        ctx.composer.enqueue(text, display, destination)
      }
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

    if (text === display && !attachments?.length) {
      ctx.transcript.send(text)
    } else {
      ctx.transcript.send(text, true, display, value => value, { attachments })
    }
  } catch (error) {
    keepEdit()

    if (!stale()) {
      ctx.transcript.sys(rewindBusyText(error, 'rewind') ?? `error: ${rpcErrorMessage(error)}`)
    }
  }
}

/** Fork from here: a NEW session holding everything before `target` (the owner's `branch` mutation,
 * the same one Desktop's message-level branch sends), switched to, with the picked prompt prefilled in
 * the composer and not sent. The original session is not touched. The first prompt has nothing
 * before it, so it forks into a plain new session with the prompt prefilled. */
export async function forkFrom(ctx: SlashRunCtx, sid: string, target: RewindTarget) {
  const prefill = (session: null | string) => {
    if (session && getUiState().sid === session) {
      prefillPrompt(ctx, target.text)
      ctx.transcript.sys(t('slashCmd.core.rewind.forked', String(target.turn)))
    }
  }

  try {
    if (!target.throughRowId) {
      await ctx.session.newSession()

      return prefill(getUiState().sid)
    }

    const stale = () => getUiState().sid !== sid

    const mutation = await mutateCanonical(
      ctx.gateway.gw,
      sid,
      'branch',
      { through_message_id: target.throughRowId },
      stale
    )

    if (!mutation?.result || stale()) {
      return
    }

    await applyBranchResult(mutation.result, '', ctx)
    prefill(mutation.result.branched_session_id as string)
  } catch (error) {
    if (getUiState().sid === sid) {
      ctx.transcript.sys(rewindBusyText(error, 'rewind') ?? `error: ${rpcErrorMessage(error)}`)
    }
  }
}
