import type { MouseTrackingMode } from '@hermes/ink'
import { useEffect, useRef } from 'react'

import { resolveDetailsMode, resolveSections } from '../domain/details.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { ConfigFullResponse, ConfigMtimeResponse, ReloadMcpResponse } from '../gatewayTypes.js'
import { normalizeLocale, translate } from '../i18n/index.js'
import { DEFAULT_VOICE_RECORD_KEY, type ParsedVoiceRecordKey, parseVoiceRecordKey } from '../lib/platform.js'
import { asRpcResult } from '../lib/rpc.js'

import { applyConfiguredTuiTheme } from './createGatewayEventHandler.js'
import {
  type BusyInputMode,
  DEFAULT_INDICATOR_STYLE,
  INDICATOR_STYLES,
  type IndicatorStyle,
  type StatusBarMode
} from './interfaces.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

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
const TRUTHY_MOUSE_ALL = new Set(['1', 'true', 'yes', 'on', 'all', 'full', 'any'])
const hasOwn = (obj: object, key: PropertyKey) => Object.prototype.hasOwnProperty.call(obj, key)

// `display.mouse_tracking` accepts boolean (`true` ⇒ all modes, `false` ⇒ off)
// for back-compat, plus the string presets `off|wheel|buttons|all` (aliases:
// `on`/`full`/`any`/`1`/`true`/... → `all`; `0`/`false`/`no`/`off` → `off`).
// `wheel` enables 1000+1006 — scroll wheel + click only, no drag or hover,
// which silences tmux's "No image in clipboard" spam over the prompt row.
// `buttons` adds 1002 so terminal-side text selection drags still register.
// Legacy `tui_mouse` is honored only if `mouse_tracking` is absent.
export const normalizeMouseTracking = (display: {
  mouse_tracking?: unknown
  tui_mouse?: unknown
}): MouseTrackingMode => {
  const raw = hasOwn(display, 'mouse_tracking') ? display.mouse_tracking : display.tui_mouse

  if (raw === false || raw === 0) {
    return 'off'
  }

  if (raw === true || raw === undefined || raw === null) {
    return 'all'
  }

  if (typeof raw === 'number') {
    return 'all'
  }

  if (typeof raw !== 'string') {
    return 'all'
  }

  const v = raw.trim().toLowerCase()

  if (FALSEY_MOUSE.has(v)) {
    return 'off'
  }

  if (TRUTHY_MOUSE_ALL.has(v)) {
    return 'all'
  }

  if (v === 'wheel' || v === 'scroll') {
    return 'wheel'
  }

  if (v === 'buttons' || v === 'button' || v === 'click') {
    return 'buttons'
  }

  return 'all'
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

// ── MCP revision handshake ───────────────────────────────────────────
//
// The poll must not ack an MCP config revision until the server confirms it
// actually LOADED it. Advancing `accepted` before the reload succeeds loses
// revisions permanently: quietRpc collapses a failed reload to null, the
// next poll sees the same mcp_rev, and the new config never applies until
// an unrelated MCP edit. So `accepted` only moves on a confirmed reload —
// to the server's loaded_rev (what discovery actually read), falling back
// to the requested rev for older gateways. Retries are decoupled from
// mtime: every poll re-compares, so a transiently broken server heals on
// the next tick.

export interface McpRevState {
  /** Last revision the server CONFIRMED it loaded (or boot baseline). */
  accepted: string
  /** A reload RPC is outstanding — don't stack another every poll tick. */
  inFlight: boolean
}

export const syncMcpReload = async (
  gw: GatewayClient,
  sid: string,
  nextMcpRev: string,
  state: McpRevState,
  onReloaded?: () => void
): Promise<void> => {
  if (!nextMcpRev || nextMcpRev === state.accepted || state.inFlight) {
    return
  }

  state.inFlight = true

  try {
    const r = await quietRpc<ReloadMcpResponse>(gw, 'reload.mcp', {
      confirm: true,
      rev: nextMcpRev,
      session_id: sid
    })

    if (r?.status === 'reloaded') {
      state.accepted = String(r.loaded_rev || nextMcpRev)
      onReloaded?.()
    }
    // Failure (null) or confirm_required: leave `accepted` unchanged so the
    // next poll tick retries the same revision.
  } finally {
    state.inFlight = false
  }
}

const _voiceRecordKeyFromConfig = (cfg: ConfigFullResponse | null): ParsedVoiceRecordKey => {
  const raw = cfg?.config?.voice?.record_key

  return raw ? parseVoiceRecordKey(raw) : DEFAULT_VOICE_RECORD_KEY
}

const _pasteCollapseLinesFromConfig = (cfg: ConfigFullResponse | null): number => {
  if (!cfg?.config) {
    return 5
  }

  const raw = cfg.config.paste_collapse_threshold

  if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 0) {
    return Math.round(raw)
  }

  if (typeof raw === 'string') {
    const n = parseInt(raw, 10)

    if (Number.isFinite(n) && n >= 0) {
      return n
    }
  }

  return 5
}

const _pasteCollapseCharsFromConfig = (cfg: ConfigFullResponse | null): number => {
  if (!cfg?.config) {
    return 2000
  }

  const raw = cfg.config.paste_collapse_char_threshold

  if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 0) {
    return Math.round(raw)
  }

  if (typeof raw === 'string') {
    const n = parseInt(raw, 10)

    if (Number.isFinite(n) && n >= 0) {
      return n
    }
  }

  return 2000
}

/** Fetch ``config.get full`` and fan the result through ``applyDisplay``.
 *
 * Both initial hydration and live config refresh use this helper. Keeping
 * display hydration separate from MCP reload is load-bearing: rebuilding the
 * tool schema after a turn has started invalidates the prompt cache. */
export async function hydrateFullConfig(
  gw: GatewayClient,
  setBell: (v: boolean) => void,
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void
): Promise<ConfigFullResponse | null> {
  const cfg = await quietRpc<ConfigFullResponse>(gw, 'config.get', { key: 'full' })

  // A transient read failure must preserve every last-good display value,
  // not only locale and voice.record_key. The mtime poll deliberately keeps
  // the previous revision in this case so the same edit is retried.
  if (cfg) {
    applyDisplay(cfg, setBell, setVoiceRecordKey)
  }

  return cfg
}

/** Apply a live config-file change without rebuilding the agent tool schema.
 *
 * MCP reloads belong to the independent ``mcp_rev`` handshake (or the
 * explicit slash command). Display-only changes — most notably
 * ``display.language`` — must never pay that prompt-cache cost merely because
 * they share ``config.yaml`` with MCP settings.
 */
export const syncChangedConfig = hydrateFullConfig

/** Refresh a changed config revision and acknowledge it only after the full
 * config payload was applied successfully. Keeping the old revision on failure
 * makes the next poll retry the same edit instead of waiting for another write. */
export async function syncConfigRevision(
  gw: GatewayClient,
  previousMtime: number,
  setBell: (v: boolean) => void,
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void,
  observedRevision?: ConfigMtimeResponse | null
): Promise<number> {
  const revision =
    observedRevision === undefined
      ? await quietRpc<ConfigMtimeResponse>(gw, 'config.get', { key: 'mtime' })
      : observedRevision

  const next = Number(revision?.mtime ?? 0)

  if (!next || next === previousMtime) {
    return previousMtime
  }

  const cfg = await syncChangedConfig(gw, setBell, setVoiceRecordKey)

  return cfg ? next : previousMtime
}

export const applyDisplay = (
  cfg: ConfigFullResponse | null,
  setBell: (v: boolean) => void,
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void
) => {
  if (!cfg) {
    return
  }

  const d = cfg?.config?.display ?? {}

  setBell(!!d.bell_on_complete)

  applyConfiguredTuiTheme(d.tui_theme)

  if (setVoiceRecordKey) {
    setVoiceRecordKey(_voiceRecordKeyFromConfig(cfg))
  }

  patchUiState({
    battery: !!d.battery,
    busyInputMode: normalizeBusyInputMode(d.busy_input_mode),
    compact: !!d.tui_compact,
    detailsMode: resolveDetailsMode(d),
    detailsModeCommandOverride: false,
    indicatorStyle: normalizeIndicatorStyle(d.tui_status_indicator),
    inlineDiffs: d.inline_diffs !== false,
    locale: normalizeLocale(d.language),
    mouseTracking: normalizeMouseTracking(d),
    pasteCollapseLines: _pasteCollapseLinesFromConfig(cfg),
    pasteCollapseChars: _pasteCollapseCharsFromConfig(cfg),
    sections: resolveSections(d.sections),
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
  const syncInFlightRef = useRef(false)
  const mcpRevRef = useRef<McpRevState>({ accepted: '', inFlight: false })

  useEffect(() => {
    if (!sid) {
      return
    }

    // Keep startup cheap: voice.toggle status probes optional audio/STT deps and
    // can run long enough to delay prompt.submit on the single stdio RPC pipe.
    // Environment flags are enough to initialize the UI bit; the heavier status
    // check still runs when the user opens /voice.
    mtimeRef.current = 0
    mcpRevRef.current = { accepted: '', inFlight: false }
    setVoiceEnabled(process.env.HERMES_VOICE === '1')
    let active = true
    syncInFlightRef.current = true
    void (async () => {
      const revision = await quietRpc<ConfigMtimeResponse>(gw, 'config.get', { key: 'mtime' })
      const cfg = await hydrateFullConfig(gw, setBellOnComplete, setVoiceRecordKey)

      if (active) {
        mcpRevRef.current.accepted = String(revision?.mcp_rev ?? '')

        if (cfg) {
          mtimeRef.current = Number(revision?.mtime ?? 0)
        }
      }
    })().finally(() => {
      if (active) {
        syncInFlightRef.current = false
      }
    })

    return () => {
      active = false
      syncInFlightRef.current = false
    }
  }, [gw, setBellOnComplete, setVoiceEnabled, setVoiceRecordKey, sid])

  useEffect(() => {
    if (!sid) {
      return
    }

    let active = true

    const id = setInterval(() => {
      if (syncInFlightRef.current) {
        return
      }

      syncInFlightRef.current = true
      void (async () => {
        const revision = await quietRpc<ConfigMtimeResponse>(gw, 'config.get', { key: 'mtime' })

        if (!active || !revision) {
          return
        }

        const nextMcpRev = String(revision.mcp_rev ?? '')

        if (!mtimeRef.current && nextMcpRev && !mcpRevRef.current.accepted) {
          // Seed the baseline after a transient startup read failure. The
          // current revision is already loaded by the gateway, so it must not
          // be mistaken for a new MCP edit.
          mcpRevRef.current.accepted = nextMcpRev
        } else if (nextMcpRev) {
          void syncMcpReload(gw, sid, nextMcpRev, mcpRevRef.current, () =>
            turnController.pushActivity(
              translate(getUiState().locale, 'activity.mcpReloadedAfterConfigChange')
            )
          )
        }

        const next = await syncConfigRevision(
          gw,
          mtimeRef.current,
          setBellOnComplete,
          setVoiceRecordKey,
          revision
        )

        if (active) {
          mtimeRef.current = next
        }
      })()
        .finally(() => {
          if (active) {
            syncInFlightRef.current = false
          }
        })
    }, MTIME_POLL_MS)

    return () => {
      active = false
      clearInterval(id)
      syncInFlightRef.current = false
    }
  }, [gw, setBellOnComplete, setVoiceRecordKey, sid])
}

export interface UseConfigSyncOptions {
  gw: GatewayClient
  setBellOnComplete: (v: boolean) => void
  setVoiceEnabled: (v: boolean) => void
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void
  sid: null | string
}
