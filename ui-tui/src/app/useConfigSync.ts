import { useEffect, useRef } from 'react'

import { resolveDetailsMode, resolveSections } from '../domain/details.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  ConfigFullResponse,
  ConfigMtimeResponse,
  ReloadMcpResponse,
  VoiceToggleResponse
} from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'

import type { BusyInputMode, StatusBarMode } from './interfaces.js'
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

const BUSY_MODES = new Set<BusyInputMode>(['interrupt', 'queue', 'steer'])

// TUI defaults to `queue` even though the framework default
// (`hermes_cli/config.py`) is `interrupt`.  Rationale: in a full-screen
// TUI you're typically authoring the next prompt while the agent is
// still streaming, and an unintended interrupt loses work.  Set
// `display.busy_input_mode: interrupt` (or `steer`) explicitly to
// opt out per-config; CLI / messaging adapters keep their `interrupt`
// default unchanged.
const TUI_BUSY_DEFAULT: BusyInputMode = 'queue'

export const normalizeBusyInputMode = (raw: unknown): BusyInputMode => {
  if (typeof raw !== 'string') {
    return TUI_BUSY_DEFAULT
  }

  const v = raw.trim().toLowerCase() as BusyInputMode

  return BUSY_MODES.has(v) ? v : TUI_BUSY_DEFAULT
}

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
    busyInputMode: normalizeBusyInputMode(d.busy_input_mode),
    compact: !!d.tui_compact,
    detailsMode: resolveDetailsMode(d),
    detailsModeCommandOverride: false,
    inlineDiffs: d.inline_diffs !== false,
    mouseTracking: d.tui_mouse !== false,
    sections: resolveSections(d.sections),
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
