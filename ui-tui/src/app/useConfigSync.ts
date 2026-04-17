import { useEffect, useRef } from 'react'

import { resolveDetailsMode } from '../domain/details.js'
import type {
  ConfigFullResponse,
  ConfigMtimeResponse,
  ReloadMcpResponse,
  VoiceToggleResponse
} from '../gatewayTypes.js'

import type { GatewayRpc } from './interfaces.js'
import { turnController } from './turnController.js'
import { patchUiState } from './uiStore.js'

const MTIME_POLL_MS = 5000

const applyDisplay = (cfg: ConfigFullResponse | null, setBell: (v: boolean) => void) => {
  const d = cfg?.config?.display ?? {}

  setBell(!!d.bell_on_complete)
  patchUiState({
    compact: !!d.tui_compact,
    detailsMode: resolveDetailsMode(d),
    statusBar: d.tui_statusbar !== false
  })
}

export function useConfigSync({ rpc, setBellOnComplete, setVoiceEnabled, sid }: UseConfigSyncOptions) {
  const mtimeRef = useRef(0)

  useEffect(() => {
    if (!sid) {
      return
    }

    rpc<VoiceToggleResponse>('voice.toggle', { action: 'status' }).then(r => setVoiceEnabled(!!r?.enabled))
    rpc<ConfigMtimeResponse>('config.get', { key: 'mtime' }).then(r => {
      mtimeRef.current = Number(r?.mtime ?? 0)
    })
    rpc<ConfigFullResponse>('config.get', { key: 'full' }).then(r => applyDisplay(r, setBellOnComplete))
  }, [rpc, setBellOnComplete, setVoiceEnabled, sid])

  useEffect(() => {
    if (!sid) {
      return
    }

    const id = setInterval(() => {
      rpc<ConfigMtimeResponse>('config.get', { key: 'mtime' }).then(r => {
        const next = Number(r?.mtime ?? 0)

        if (!mtimeRef.current) {
          if (next) {
            mtimeRef.current = next
          }

          return
        }

        if (!next || next === mtimeRef.current) {
          return
        }

        mtimeRef.current = next

        rpc<ReloadMcpResponse>('reload.mcp', { session_id: sid }).then(
          r => r && turnController.pushActivity('MCP reloaded after config change')
        )
        rpc<ConfigFullResponse>('config.get', { key: 'full' }).then(r => applyDisplay(r, setBellOnComplete))
      })
    }, MTIME_POLL_MS)

    return () => clearInterval(id)
  }, [rpc, setBellOnComplete, sid])
}

export interface UseConfigSyncOptions {
  rpc: GatewayRpc
  setBellOnComplete: (v: boolean) => void
  setVoiceEnabled: (v: boolean) => void
  sid: null | string
}
