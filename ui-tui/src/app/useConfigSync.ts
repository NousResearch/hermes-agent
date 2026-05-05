import { useEffect, useRef } from 'react'

import { resolveDetailsMode, resolveSections } from '../domain/details.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  ConfigFullResponse,
  ConfigMtimeResponse,
  ReloadMcpResponse
} from '../gatewayTypes.js'
import {
  DEFAULT_VOICE_RECORD_KEY,
  parseVoiceRecordKey,
  type ParsedVoiceRecordKey
} from '../lib/platform.js'
import { asRpcResult } from '../lib/rpc.js'

import {
  type BusyInputMode,
  DEFAULT_INDICATOR_STYLE,
  INDICATOR_STYLES,
  type IndicatorStyle,
  type StatusBarMode
} from './interfaces.js'
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

const INDICATOR_STYLE_SET: ReadonlySet<IndicatorStyle> = new Set(INDICATOR_STYLES)

export const normalizeIndicatorStyle = (raw: unknown): IndicatorStyle => {
  if (typeof raw !== 'string') {
    return DEFAULT_INDICATOR_STYLE
  }

  const v = raw.trim().toLowerCase() as IndicatorStyle

  return INDICATOR_STYLE_SET.has(v) ? v : DEFAULT_INDICATOR_STYLE
}

const FALSEY_MOUSE = new Set(['0', 'false', 'no', 'off'])
const hasOwn = (obj: object, key: PropertyKey) => Object.prototype.hasOwnProperty.call(obj, key)

export const normalizeMouseTracking = (display: { mouse_tracking?: unknown; tui_mouse?: unknown }): boolean => {
  const raw = hasOwn(display, 'mouse_tracking') ? display.mouse_tracking : display.tui_mouse

  if (raw === false || raw === 0) {
    return false
  }

  return typeof raw === 'string' ? !FALSEY_MOUSE.has(raw.trim().toLowerCase()) : true
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

const _voiceRecordKeyFromConfig = (cfg: ConfigFullResponse | null): ParsedVoiceRecordKey => {
  const raw = cfg?.config?.voice?.record_key

  return raw ? parseVoiceRecordKey(raw) : DEFAULT_VOICE_RECORD_KEY
}

/** Fetch ``config.get full`` and fan the result through ``applyDisplay``.
 *
 * Extracted so the mtime-reload path can be exercised by the test
 * suite without a React runtime (Copilot round-12 review on #19835).
 * Both the initial hydration and the mtime poller use this shared
 * helper, so a regression in the fetch/apply plumbing now fails the
 * useConfigSync tests instead of only being visible at runtime. */
export async function hydrateFullConfig(
  gw: GatewayClient,
  setBell: (v: boolean) => void,
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void
): Promise<ConfigFullResponse | null> {
  const cfg = await quietRpc<ConfigFullResponse>(gw, 'config.get', { key: 'full' })
  applyDisplay(cfg, setBell, setVoiceRecordKey)
  return cfg
}

export const applyDisplay = (
  cfg: ConfigFullResponse | null,
  setBell: (v: boolean) => void,
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void
) => {
  const d = cfg?.config?.display ?? {}

  setBell(!!d.bell_on_complete)
  // Only push the voice record key when the RPC actually returned a
  // config payload. ``quietRpc()`` collapses failures to ``null``; if we
  // reset the cached shortcut on every null we would clobber a custom
  // binding after one transient RPC error until the next config edit
  // (Copilot round-8 review on #19835). The mtime-poll loop advances
  // ``mtimeRef`` before this call, so staying silent on null preserves
  // the last-good state and lets the next successful poll refresh it.
  if (setVoiceRecordKey && cfg) {
    setVoiceRecordKey(_voiceRecordKeyFromConfig(cfg))
  }
  patchUiState({
    busyInputMode: normalizeBusyInputMode(d.busy_input_mode),
    compact: !!d.tui_compact,
    detailsMode: resolveDetailsMode(d),
    detailsModeCommandOverride: false,
    indicatorStyle: normalizeIndicatorStyle(d.tui_status_indicator),
    inlineDiffs: d.inline_diffs !== false,
    mouseTracking: normalizeMouseTracking(d),
    sections: resolveSections(d.sections),
    showCost: !!d.show_cost,
    showReasoning: !!d.show_reasoning,
    statusBar: normalizeStatusBar(d.tui_statusbar),
    streaming: d.streaming !== false
  })
}

export function useConfigSync({
  gw,
  setBellOnComplete,
  setVoiceEnabled,
  setVoiceRecordKey,
  sid
}: UseConfigSyncOptions) {
  const mtimeRef = useRef(0)

  useEffect(() => {
    if (!sid) {
      return
    }

    // Keep startup cheap: voice.toggle status probes optional audio/STT deps and
    // can run long enough to delay prompt.submit on the single stdio RPC pipe.
    // Environment flags are enough to initialize the UI bit; the heavier status
    // check still runs when the user opens /voice.
    setVoiceEnabled(process.env.HERMES_VOICE === '1')
    quietRpc<ConfigMtimeResponse>(gw, 'config.get', { key: 'mtime' }).then(r => {
      mtimeRef.current = Number(r?.mtime ?? 0)
    })
    void hydrateFullConfig(gw, setBellOnComplete, setVoiceRecordKey)
  }, [gw, setBellOnComplete, setVoiceEnabled, setVoiceRecordKey, sid])

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

        quietRpc<ReloadMcpResponse>(gw, 'reload.mcp', { session_id: sid, confirm: true }).then(
          r => r && turnController.pushActivity('MCP reloaded after config change')
        )
        void hydrateFullConfig(gw, setBellOnComplete, setVoiceRecordKey)
      })
    }, MTIME_POLL_MS)

    return () => clearInterval(id)
  }, [gw, setBellOnComplete, setVoiceRecordKey, sid])
}

export interface UseConfigSyncOptions {
  gw: GatewayClient
  setBellOnComplete: (v: boolean) => void
  setVoiceEnabled: (v: boolean) => void
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void
  sid: null | string
}
