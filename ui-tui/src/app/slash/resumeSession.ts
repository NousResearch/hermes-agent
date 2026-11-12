import type { SessionListResult } from '@hermes/shared/gateway-events'

import { t } from '../../i18n/runtime.js'

import type { SlashRunCtx } from './types.js'

export function resumeSession(target: string, ctx: SlashRunCtx) {
  if (!ctx.gateway.gw.isCanonical) {
    return ctx.session.resumeById(target)
  }

  const resolve = (id: string) => {
    if (ctx.stale() || ctx.session.guardBusySessionSwitch(t('slashCmd.session.sessions.busyGuardAction'))) {
      return null
    }

    return id
  }

  // Resolve before attachment so its event hold and pending-input identity use the real ID.
  // The owner applies ID-first/title-second lookup and selects the current continuation.
  return ctx.session.resumeById(target, () =>
    ctx.gateway.gw
      .request<SessionListResult>('session.list', { title: target })
      .then(
        result => {
          if (!Array.isArray(result?.sessions)) {
            throw new Error(t('session.common.invalidResponse', 'session.list'))
          }

          const row = result.sessions[0]

          return resolve(row?.resolved_id || row?.id || target)
        },
        error => {
          // Messaging IDs and abbreviated local IDs need the normal ID-resume path.
          if ((error as { data?: { reason?: string } } | null)?.data?.reason === 'not_found') {
            return resolve(target)
          }

          throw error
        }
      )
      .catch(error => {
        ctx.guardedErr(error)

        return null
      })
  )
}
