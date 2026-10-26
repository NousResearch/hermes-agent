import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useEffect } from 'react'

import type { CommandsCatalogResponse } from '../gatewayTypes.js'
import { t } from '../i18n/runtime.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { SlashCatalog } from '../types.js'

import type { GatewayRpc } from './interfaces.js'
import { turnController } from './turnController.js'
import { $uiState } from './uiStore.js'

/** `commands.catalog` bound to `sid` when there is one: project-local skills follow the session's repo,
 *  so the catalog is reloaded whenever the live session changes. A reply for a session that is no longer
 *  the live one is dropped. */
export const loadSlashCatalog = (
  rpc: GatewayRpc,
  sid: null | string,
  setCatalog: (catalog: null | SlashCatalog) => void
) =>
  rpc<CommandsCatalogResponse>('commands.catalog', sid ? { session_id: sid } : {})
    .then(r => {
      if (!r?.pairs || $uiState.get().sid !== sid) {
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
    .catch(e =>
      turnController.pushActivity(t('gatewayMsg.startup.commandCatalogUnavailable', rpcErrorMessage(e)), 'info')
    )

/** `skills.changed`: a skill was installed or removed from another terminal. Refetch the palette
 *  from the gateway, which rescans; this client never reads the skills dir itself. True when handled. */
export function refreshSlashCatalogOnSkillsChange(
  ev: { type: string },
  rpc: GatewayRpc,
  setCatalog: (catalog: null | SlashCatalog) => void
): boolean {
  if (ev.type !== 'skills.changed') {
    return false
  }

  void loadSlashCatalog(rpc, $uiState.get().sid, setCatalog)

  return true
}

/** The slash catalog for the live session (or the launch workspace before one exists). */
export const $slashCatalog = atom<null | SlashCatalog>(null)

export const setSlashCatalog = (next: null | SlashCatalog) => $slashCatalog.set(next)

/** The slash catalog, reloaded for each live session while connected; the gateway-ready load covers the
 *  pre-session one. */
export function useSlashCatalog(rpc: GatewayRpc) {
  const catalog = useStore($slashCatalog)
  const { gatewayConnected, sid } = useStore($uiState)

  useEffect(() => {
    if (sid && gatewayConnected !== false) {
      void loadSlashCatalog(rpc, sid, setSlashCatalog)
    }
  }, [rpc, sid, gatewayConnected])

  return catalog
}
