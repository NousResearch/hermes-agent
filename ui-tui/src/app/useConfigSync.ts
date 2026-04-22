import { useEffect, useRef } from 'react'

import { resolveDetailsMode } from '../domain/details.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  ConfigFullResponse,
  ConfigMtimeResponse,
  ReloadMcpResponse,
  VoiceToggleResponse
} from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'

import type { StatusBarMode } from './interfaces.js'
import { turnController } from './turnController.js'
import { patchUiState } from './uiStore.js'

const STATUSBAR_ALIAS: Record<string, StatusBarMode> = {
  bottom: 'bottom',
  off: 'off',
  on: 'top',
  top: 'top'
}

export const normalizeStatusBar = (raw: unknown): StatusBarMode =>
  raw === false ? 'off' : typeof raw === 'string' ? (STATUSBAR_ALIAS[raw.trim().toLowerCase()] ?? 'top') : 'top'

const MTIME_POLL_MS = 5000

const quietRpc = async <T extends Record<string, any> = Record<string, any>>(
  gw: GatewayClient,
  method: string,
  params: Record<string, unknown> = {}
): Promise<null | T> => {
  try {
    return asRpcResult<T>(await gw.request<T>(method, params))
  } catch {
    return null
  }
}

export const applyDisplay = (cfg: ConfigFullResponse | null, setBell: (v: boolean) => void) => {
  const d = cfg?.config?.display ?? {}

  setBell(!!d.bell_on_complete)
  patchUiState({
    compact: !!d.tui_compact,
    detailsMode: resolveDetailsMode(d),
    inlineDiffs: d.inline_diffs !== false,
    showCost: !!d.show_cost,
    showReasoning: !!d.show_reasoning,
    statusBar: normalizeStatusBar(d.tui_statusbar),
    streaming: d.streaming !== false
  })
}

export function useConfigSync({ gw, setBellOnComplete, setVoiceEnabled, sid }: UseConfigSyncOptions) {
  const mtimeRef = useRef(0)

  useEffect(() => {
    if (!sid) {
      return
    }

    quietRpc<VoiceToggleResponse>(gw, 'voice.toggle', { action: 'status' }).then(r => setVoiceEnabled(!!r?.enabled))
    quietRpc<ConfigMtimeResponse>(gw, 'config.get', { key: 'mtime' }).then(r => {
      mtimeRef.current = Number(r?.mtime ?? 0)
    })
    quietRpc<ConfigFullResponse>(gw, 'config.get', { key: 'full' }).then(r => applyDisplay(r, setBellOnComplete))
  }, [gw, setBellOnComplete, setVoiceEnabled, sid])

  useEffect(() => {
    if (!sid) {
      return
    }

    const id = setInterval(() => {
      quietRpc<ConfigMtimeResponse>(gw, 'config.get', { key: 'mtime' }).then(r => {
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

        quietRpc<ReloadMcpResponse>(gw, 'reload.mcp', { session_id: sid }).then(
          r => r && turnController.pushActivity('MCP reloaded after config change')
        )
        quietRpc<ConfigFullResponse>(gw, 'config.get', { key: 'full' }).then(r => applyDisplay(r, setBellOnComplete))
      })
    }, MTIME_POLL_MS)

    return () => clearInterval(id)
  }, [gw, setBellOnComplete, sid])
}

export interface UseConfigSyncOptions {
  gw: GatewayClient
  setBellOnComplete: (v: boolean) => void
  setVoiceEnabled: (v: boolean) => void
  sid: null | string
}
