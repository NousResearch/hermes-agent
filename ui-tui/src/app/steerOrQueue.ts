import type { SessionSteerResponse } from '../gatewayTypes.js'
import { t } from '../i18n/runtime.js'

import type { ComposerActions, GatewayRpc } from './interfaces.js'
import { isCurrentDestination, type SubmissionDestination } from './submissionDestination.js'

const PREVIEW_CHARS = 50

const previewOf = (text: string) => `${text.slice(0, PREVIEW_CHARS)}${text.length > PREVIEW_CHARS ? '…' : ''}`

/** Steer `text` into the running turn (`session.steer`); when the owner refuses it (no active turn, the
 * turn ended first) or the call fails, the words are queued as the next prompt instead of lost.
 *
 * `destination` is captured before the await and owns both outcomes: the fallback queues into THAT
 * session (not whichever one is selected when the reply lands), and the status line is only printed
 * while it is still the one on screen. */
export async function steerOrQueue(opts: {
  destination: SubmissionDestination & { sid: string }
  enqueue: ComposerActions['enqueue']
  generation?: number
  rpc: GatewayRpc
  sys: (text: string) => void
  text: string
}): Promise<void> {
  const { destination, enqueue, generation, rpc, sys, text } = opts

  const result = await rpc<SessionSteerResponse>('session.steer', {
    session_id: destination.sid,
    text,
    ...(generation !== undefined ? { execution_generation: generation } : {})
  }).catch(() => null)

  const status = (line: string) => {
    if (isCurrentDestination(destination)) {
      sys(line)
    }
  }

  if (result?.status === 'queued') {
    return status(t('slashCmd.core.steer.queued', previewOf(text)))
  }

  enqueue(text, text, destination)
  status(t('slashCmd.core.steer.rejected'))
}
