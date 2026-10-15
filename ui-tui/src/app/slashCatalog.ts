import type { CommandsCatalogResponse } from '../gatewayTypes.js'
import { t } from '../i18n/runtime.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { SlashCatalog } from '../types.js'

import type { GatewayRpc, StateSetter } from './interfaces.js'
import { turnController } from './turnController.js'

/** `commands.catalog` bound to `sid` when there is one: project-local skills follow the session's repo,
 *  so the catalog is reloaded whenever the live session changes. */
export const loadSlashCatalog = (rpc: GatewayRpc, sid: null | string, setCatalog: StateSetter<null | SlashCatalog>) =>
  rpc<CommandsCatalogResponse>('commands.catalog', sid ? { session_id: sid } : {})
    .then(r => {
      if (!r?.pairs) {
        return
      }

      setCatalog({
        canon: r.canon ?? {},
        categories: r.categories ?? [],
        pairs: r.pairs,
        skillCount: r.skill_count ?? 0,
        sub: r.sub ?? {}
      })

      if (r.warning) {
        turnController.pushActivity(r.warning, 'warn')
      }
    })
    .catch(e => turnController.pushActivity(t('gatewayMsg.startup.commandCatalogUnavailable', rpcErrorMessage(e)), 'info'))
