import { useCallback, useMemo } from 'react'

import { sharedControlParams } from '../canonicalGateway.js'
import type { SharedControlRespondResponse } from '../gatewayTypes.js'
import { t } from '../i18n/runtime.js'

import type { ComposerActions, GatewayRpc, OverlayState } from './interfaces.js'
import { capturePromptResponseGuard, patchOverlayState } from './overlayStore.js'
import { steerOrQueue } from './steerOrQueue.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

/** The approval prompt's answers: a gateway choice, or "No, and tell Hermes what to do instead". */
export function useApprovalAnswers(
  approval: OverlayState['approval'],
  canonical: boolean,
  enqueue: ComposerActions['enqueue'],
  respondWith: (requestId: string, result: Record<string, unknown>, done: () => void) => void,
  rpc: GatewayRpc,
  sys: (text: string) => void
) {
  const answerApproval = useCallback(
    (choice: string) => {
      const fresh = capturePromptResponseGuard('approval', approval)

      if (!fresh() || !approval) {
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
    },
    [approval, respondWith, rpc]
  )

  // "No, and tell Hermes what to do instead": the instruction is steered into the held turn FIRST, so it
  // rides on the blocked tool result; then the command is denied. A refused steer (the turn is
  // ending) keeps the words as the next prompt, the same fallback /steer uses.
  const answerApprovalTell = useCallback(
    (text: string) => {
      const sid = getUiState().sid

      if (!sid || !capturePromptResponseGuard('approval', approval)()) {
        return
      }

      void steerOrQueue({
        enqueue: enqueue,
        generation: canonical ? getUiState().info?.execution_generation : undefined,
        rpc,
        sid,
        sys,
        text
      }).finally(() => answerApproval('deny'))
    },
    [answerApproval, approval, canonical, enqueue, rpc, sys]
  )

  return useMemo(() => ({ answerApproval, answerApprovalTell }), [answerApproval, answerApprovalTell])
}
