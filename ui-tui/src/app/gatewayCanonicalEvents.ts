import type { AnyGatewayEvent } from '../gatewayTypes.js'
import type { SessionInfo } from '../types.js'

import { patchOverlayState } from './overlayStore.js'
import { patchUiState } from './uiStore.js'

// Canonical-gateway event handling extracted from createGatewayEventHandler:
// the owner-epoch lifecycle fence and the generation-bound shared controls
// (`gateway/session_pending_controls.py`).

type GatewayEventOf<K extends AnyGatewayEvent['type']> = Extract<AnyGatewayEvent, { type: K }>

interface ExecutionStamp {
  execution_epoch?: string
  execution_generation?: number
}

/** An `error` event carrying no execution stamp (build/RPC failure, not a turn settlement). */
export const isGenericErrorEvent = (ev: AnyGatewayEvent): boolean => {
  const execution = (ev.payload ?? {}) as ExecutionStamp

  return ev.type === 'error' && execution.execution_epoch === undefined && execution.execution_generation === undefined
}

/** Lifecycle authority is local to an owner epoch. Only attachment RPC
 *  snapshots replace that epoch; delayed push events cannot reset it.
 *  Returns true when the event must be dropped. */
export function fenceLifecycleEvent(ev: AnyGatewayEvent, current: null | SessionInfo, genericError: boolean): boolean {
  const lifecycle = !genericError && ['session.info', 'message.start', 'message.complete', 'error'].includes(ev.type)

  if (lifecycle && current?.execution_generation !== undefined) {
    const incoming = (ev.payload ?? {}) as ExecutionStamp

    if (
      incoming.execution_epoch !== current.execution_epoch ||
      typeof incoming.execution_generation !== 'number' ||
      !Number.isSafeInteger(incoming.execution_generation) ||
      incoming.execution_generation < current.execution_generation
    ) {
      return true
    }

    if (ev.type !== 'session.info') {
      patchUiState({ info: { ...current, execution_generation: incoming.execution_generation } })
    }
  }

  return false
}

/** Clears the shared approval/clarify card a settled event names. Returns true when handled. */
export function settleSharedPrompt(ev: AnyGatewayEvent): boolean {
  if (ev.type === 'approval.settled' || ev.type === 'clarify.settled') {
    const kind = ev.type === 'approval.settled' ? 'approval' : 'clarify'
    const promptId = ev.payload?.prompt_id

    patchOverlayState(previous =>
      previous[kind]?.sharedControl?.prompt_id === promptId ? { ...previous, [kind]: null } : previous
    )

    return true
  }

  return false
}

// Canonical gateways (`gateway/session_pending_controls.py`) publish
// generation-bound shared controls as events carrying `prompt_id`; they
// are answered through approval.respond / clarify.respond, never through
// a server→client request frame (that is the legacy tui_gateway path in
// createServerRequestHandler).
export function openCanonicalClarify(
  ev: GatewayEventOf<'clarify.request'>,
  setStatus: (status: string) => void,
  ringPromptBell: () => void
): void {
  const shared = ev.payload

  if (!shared?.prompt_id || !ev.session_id || typeof shared.execution_generation !== 'number') {
    return
  }

  const sharedControl = {
    session_id: ev.session_id,
    execution_generation: shared.execution_generation,
    prompt_id: shared.prompt_id
  }

  const batch = (shared.questions ?? [])
    .filter(q => typeof q?.qid === 'string' && q.qid && typeof q?.question === 'string' && q.question.trim())
    .map(q => ({
      choices: q.choices && q.choices.length > 0 ? q.choices : null,
      multiSelect: q.multi_select === true,
      qid: q.qid,
      question: q.question.trim()
    }))

  // The canonical gateway asks one card per question (question/choices/
  // multi_select); the Ink card renders the one questions[] shape, so a
  // single card becomes a one-question set answered via clarify.respond.
  const single = (shared.question ?? '').trim()

  const questions = batch.length
    ? batch
    : single
      ? [
          {
            choices: shared.choices && shared.choices.length > 0 ? shared.choices : null,
            multiSelect: shared.multi_select === true,
            qid: shared.prompt_id,
            question: single
          }
        ]
      : []

  if (!questions.length) {
    return
  }

  patchOverlayState({
    clarify: { answers: shared.answers ?? {}, questions, requestId: shared.prompt_id, sharedControl }
  })
  setStatus('waiting for input…')
  ringPromptBell()
}

export function openCanonicalApproval(
  ev: GatewayEventOf<'approval.request'>,
  setStatus: (status: string) => void,
  ringPromptBell: () => void
): void {
  const shared = ev.payload

  if (!shared?.prompt_id || !ev.session_id || typeof shared.execution_generation !== 'number') {
    return
  }

  const sharedControl = {
    session_id: ev.session_id,
    execution_generation: shared.execution_generation,
    prompt_id: shared.prompt_id
  }

  patchOverlayState({
    approval: {
      // Only an explicit false (tirith warning) drops the permanent-allow option.
      allowPermanent: shared.allow_permanent !== false,
      choices: shared.choices,
      command: String(shared.command ?? ''),
      description: String(shared.description ?? 'dangerous command'),
      requestId: shared.prompt_id,
      sharedControl,
      smartDenied: shared.smart_denied === true
    }
  })
  setStatus('approval needed')
  ringPromptBell()
}
