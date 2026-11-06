import { useMemo } from 'react'

import { sharedControlParams } from '../canonicalGateway.js'
import type { SharedControlRespondResponse } from '../gatewayTypes.js'
import { t } from '../i18n/runtime.js'

import type { ComposerActions, GatewayRpc, OverlayState } from './interfaces.js'
import { capturePromptResponseGuard, patchOverlayState } from './overlayStore.js'
import { steerOrQueue } from './steerOrQueue.js'
import { captureDestination } from './submissionDestination.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

interface ApprovalAnswerDeps {
  approval: OverlayState['approval']
  canonical: boolean
  enqueue: ComposerActions['enqueue']
  respondWith: (requestId: string, result: Record<string, unknown>, done: () => void) => void
  rpc: GatewayRpc
  sys: (text: string) => void
}

// An approval answered with "tell Hermes instead" is committed to a deny. While its steer is in flight
// ('steering') every answer is refused: a repeated Enter would steer twice and an allow would release
// the command. Once steered ('told') only the promised deny may still be sent (a retry after a failed
// respond). Keyed by the approval object, so a new prompt starts clean.
const tellState = new WeakMap<object, 'steering' | 'told'>()

/** The approval prompt's answers: a gateway choice, or "No, and tell Hermes what to do instead". */
export function approvalAnswers({ approval, canonical, enqueue, respondWith, rpc, sys }: ApprovalAnswerDeps) {
  const answerApproval = (choice: string) => {
    const fresh = capturePromptResponseGuard('approval', approval)
    const told = approval ? tellState.get(approval) : undefined

    if (!fresh() || !approval || told === 'steering' || (told && choice !== 'deny')) {
      return
    }

    const settle = () => {
      if (!fresh()) {
        return
      }

      patchOverlayState({ approval: null })
      patchTurnState({
        outcome: choice === 'deny' ? t('session.approval.denied') : t('session.approval.approved', choice)
      })
      patchUiState({ status: 'running…' })
    }

    // Canonical shared controls answer through the generation-bound RPC; a
    // legacy server→client request resolves its response frame locally.
    if (approval.sharedControl) {
      return rpc<SharedControlRespondResponse>('approval.respond', {
        choice,
        ...sharedControlParams(approval)
      }).then(r => r && settle())
    }

    respondWith(approval.requestId, { choice }, settle)
  }

  // "No, and tell Hermes what to do instead": the instruction is steered into the held turn FIRST, so it
  // rides on the blocked tool result; then the command is denied. A refused steer (the turn is
  // ending) keeps the words as the next prompt, the same fallback /steer uses.
  const answerApprovalTell = (text: string) => {
    const destination = captureDestination()
    const sid = destination.sid

    if (!sid || !approval || tellState.has(approval) || !capturePromptResponseGuard('approval', approval)()) {
      return
    }

    tellState.set(approval, 'steering')
    void steerOrQueue({
      destination: { ...destination, sid },
      enqueue,
      generation: canonical ? getUiState().info?.execution_generation : undefined,
      rpc,
      sys,
      text
    }).finally(() => {
      tellState.set(approval, 'told')
      answerApproval('deny')
    })
  }

  return { answerApproval, answerApprovalTell }
}

export function useApprovalAnswers(
  approval: OverlayState['approval'],
  canonical: boolean,
  enqueue: ComposerActions['enqueue'],
  respondWith: ApprovalAnswerDeps['respondWith'],
  rpc: GatewayRpc,
  sys: (text: string) => void
) {
  return useMemo(
    () => approvalAnswers({ approval, canonical, enqueue, respondWith, rpc, sys }),
    [approval, canonical, enqueue, respondWith, rpc, sys]
  )
}
