import { useCallback, useEffect, useRef, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { asRpcResult } from '../lib/rpc.js'

/** Ghost-text reply suggestion for the empty composer (slice 1).
 *
 *  When an agent turn completes (blocked flips true -> false) and the
 *  composer is empty, ask the gateway's deterministic `complete.suggest`
 *  for a likely reply and surface the first candidate as ghost text in the
 *  placeholder slot. Tab accepts it into real input; Esc dismisses it until
 *  the next turn; typing clears it. Suggestions are decorative: any RPC
 *  failure resolves to "no ghost", never an error.
 */

export interface SuggestCandidate {
  kind: string
  text: string
}

export interface SuggestResponse {
  candidates?: SuggestCandidate[]
  history_version?: number
}

/** First candidate text from a raw RPC envelope, '' when there is none. */
export function firstCandidateText(raw: unknown): string {
  const r = asRpcResult<SuggestResponse>(raw)
  const text = r?.candidates?.[0]?.text

  return typeof text === 'string' ? text.trim() : ''
}

/** Whether the ghost should render right now. Pure so tests can pin the
 *  contract: only in an idle session, only in an EMPTY composer, only while
 *  not dismissed, and only when there is something to show. */
export function ghostVisible(opts: {
  blocked: boolean
  dismissed: boolean
  ghost: string
  input: string
}): boolean {
  return opts.input === '' && !opts.blocked && !opts.dismissed && opts.ghost !== ''
}

export function useGhostSuggestion(
  input: string,
  blocked: boolean,
  gw: GatewayClient,
  getSessionId: () => null | string | undefined
) {
  const [ghost, setGhost] = useState('')
  const [dismissed, setDismissed] = useState(false)
  const prevBlockedRef = useRef(blocked)
  const seqRef = useRef(0)

  useEffect(() => {
    const wasBlocked = prevBlockedRef.current

    prevBlockedRef.current = blocked

    // Fire exactly once per completed turn: on the busy -> idle transition.
    if (!wasBlocked || blocked) {
      return
    }

    const sid = getSessionId()

    if (!sid) {
      return
    }

    const seq = ++seqRef.current

    setDismissed(false)
    setGhost('')

    gw.request<SuggestResponse>('complete.suggest', { session_id: sid })
      .then(raw => {
        if (seq !== seqRef.current || getSessionId() !== sid) {
          return
        }

        setGhost(firstCandidateText(raw))
      })
      .catch(() => {
        // Decorative feature: failures must never reach the composer.
      })
  }, [blocked, getSessionId, gw])

  useEffect(() => {
    if (input !== '') {
      // Invalidate both a shown suggestion and any request that was started
      // while the composer was empty. Deleting the text must not resurrect it.
      seqRef.current += 1
      setGhost('')
    }
  }, [input])

  const acceptGhost = useCallback((): null | string => {
    if (!ghostVisible({ blocked, dismissed, ghost, input })) {
      return null
    }

    setGhost('')

    return ghost
  }, [blocked, dismissed, ghost, input])

  const dismissGhost = useCallback((): boolean => {
    if (!ghostVisible({ blocked, dismissed, ghost, input })) {
      return false
    }

    setDismissed(true)

    return true
  }, [blocked, dismissed, ghost, input])

  return {
    acceptGhost,
    dismissGhost,
    ghost: ghostVisible({ blocked, dismissed, ghost, input }) ? ghost : ''
  }
}
