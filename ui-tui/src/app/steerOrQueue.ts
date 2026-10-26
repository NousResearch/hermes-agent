import type { SessionSteerResponse } from '../gatewayTypes.js'
import type { QueueItem } from '../hooks/useQueue.js'
import { t } from '../i18n/runtime.js'

import type { GatewayRpc } from './interfaces.js'

const PREVIEW_CHARS = 50

const previewOf = (text: string) => `${text.slice(0, PREVIEW_CHARS)}${text.length > PREVIEW_CHARS ? '…' : ''}`

/** Steer `text` into the running turn (`session.steer`); when the owner refuses it (no active turn, the
 * turn ended first) or the call fails, the words are queued as the next prompt instead of lost. */
export async function steerOrQueue(opts: {
  enqueue: (text: string) => QueueItem | void
  generation?: number
  rpc: GatewayRpc
  sid: string
  sys: (text: string) => void
  text: string
}): Promise<void> {
  const { enqueue, generation, rpc, sid, sys, text } = opts

  const result = await rpc<SessionSteerResponse>('session.steer', {
    session_id: sid,
    text,
    ...(generation !== undefined ? { execution_generation: generation } : {})
  }).catch(() => null)

  if (result?.status === 'queued') {
    return sys(t('slashCmd.core.steer.queued', previewOf(text)))
  }

  enqueue(text)
  sys(t('slashCmd.core.steer.rejected'))
}
