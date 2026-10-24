import { mutateCanonical, type MutationSnapshot } from '../app/slash/canonicalSessionControls.js'
import type { GatewayClient } from '../gatewayClient.js'
import { asRpcResult } from '../lib/rpc.js'
import { type SessionExportPayload, writeSessionExport } from '../lib/sessionExport.js'

/** One picker-owned adapter. Keep canonical retry/retirement keyed by the real gateway,
 * observing its CAS read through the public onRead seam without another resume or proxy. */
export function switcherSessionActions(gw: GatewayClient, keepView: (id: string) => boolean, stale: () => boolean) {
  const lease: { snapshot?: MutationSnapshot } = {}
  let pending = false
  const onRead = (snapshot: MutationSnapshot) => {
    lease.snapshot = snapshot
  }

  async function withAttachment<T>(id: string, operation: () => Promise<T>) {
    if (pending) {
      throw new Error('Session action already pending')
    }
    pending = true
    const alreadyHeld = keepView(id)
    lease.snapshot = undefined
    try {
      if (!gw.isCanonical) {
        throw new Error('Selected-session actions require the canonical gateway')
      }
      return await operation()
    } finally {
      // The awaited operation can populate this through the adapter's resume callback.
      const snapshot = lease.snapshot as MutationSnapshot | undefined
      lease.snapshot = undefined
      const resolved = snapshot?.session_id || id
      try {
        if (snapshot?.subscription_id && !alreadyHeld && !keepView(resolved)) {
          await gw.request('session.detach', { session_id: resolved, subscription_id: snapshot.subscription_id })
        }
      } finally {
        pending = false
      }
    }
  }

  return {
    rename: (id: string, title: string) =>
      withAttachment(id, async () => {
        const mutation = await mutateCanonical(gw, id, 'rename', { title }, stale, onRead)
        return mutation?.result ? String(mutation.result.title ?? title).trim() : undefined
      }),
    export: (id: string) =>
      withAttachment(id, async () => {
        let resolved = id
        if (!keepView(id)) {
          const snapshot = asRpcResult<MutationSnapshot>(await gw.request('session.resume', { session_id: id }))
          lease.snapshot = snapshot ?? undefined
          if (!snapshot?.session_id || !snapshot.subscription_id) {
            throw new Error('Invalid session.resume response')
          }
          resolved = snapshot.session_id
        }
        if (stale()) {
          return
        }
        const payload = asRpcResult<SessionExportPayload>(
          await gw.request('session.export', {
            session_id: resolved,
            format: 'md',
            redact: false
          })
        )
        if (stale()) {
          return
        }
        if (typeof payload?.content !== 'string' || payload.format !== 'md') {
          throw new Error('Invalid session.export response')
        }
        return writeSessionExport(payload)
      })
  }
}
