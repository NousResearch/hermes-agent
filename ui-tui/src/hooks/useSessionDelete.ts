import type { SessionListRow } from '@hermes/shared/gateway-events'
import { useCallback, useRef, useState } from 'react'

import { mutateCanonical } from '../app/slash/canonicalSessionControls.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionDeleteResponse } from '../gatewayTypes.js'
import { messages } from '../i18n/runtime.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'

/** The picker's history delete: one request at a time, and a refused delete hands back the viewer
 *  attachment its revision read made unless `keepView` says this view already held the session.
 *  Settles through `onDeleted` or `onError`. */
export function useSessionDelete(
  gw: GatewayClient,
  onDeleted: (id: string) => void,
  onError: (message: string) => void,
  keepView: (resolvedId: string) => boolean
) {
  const [deleting, setDeleting] = useState(false)
  // Set synchronously (one stdin chunk can carry the confirming key and a click before a re-render);
  // the state guard covers a settle until the updated list renders.
  const deletingRef = useRef(false)

  const settle = () => {
    deletingRef.current = false
    setDeleting(false)
  }

  const performDelete = useCallback(
    (target: SessionListRow | undefined) => {
      if (!target || deletingRef.current) {
        return
      }

      deletingRef.current = true
      setDeleting(true)
      onError('')
      // The shared gateway has no `session.delete`: its delete is the revision-fenced
      // `session.mutate operation=delete` (the whole lineage), as the classic CLI sends it.
      let release: undefined | { session_id: string; subscription_id: string }

      const deleted = gw.isCanonical
        ? mutateCanonical(gw, target.id, 'delete', {}, undefined, snapshot => {
            // The owner subscribes under the id it resolved the read to, and reuses a subscription
            // this view already holds: release only an attachment the read itself made.
            const resolved = snapshot.session_id || target.id

            release =
              snapshot.subscription_id && !keepView(resolved)
                ? { session_id: resolved, subscription_id: snapshot.subscription_id }
                : undefined
          }).then(mutation => Boolean(mutation?.result))
        : gw
            .request<SessionDeleteResponse>('session.delete', { session_id: target.id })
            .then(raw => asRpcResult<SessionDeleteResponse>(raw)?.deleted === target.id)

      deleted
        .then(ok => {
          if (ok) {
            onDeleted(target.id)
          } else {
            onError(messages().pickers.common.invalidResponse(gw.isCanonical ? 'session.mutate' : 'session.delete'))
          }

          settle()
        })
        .catch(rejection => {
          // A refused delete hands the read's attachment back so the closed session does not reappear
          // in the live section. A plain detach: an end reason would run the owner's viewer close.
          if (release) {
            void gw.request('session.detach', release).catch(() => undefined)
          }

          onError(rpcErrorMessage(rejection))
          settle()
        })
    },
    [gw, keepView, onDeleted, onError]
  )

  const pending = useCallback(() => deleting || deletingRef.current, [deleting])

  return { deletePending: pending, deleting, performDelete }
}
