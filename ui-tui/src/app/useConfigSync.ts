import type { MouseTrackingMode } from '@hermes/ink'
import { useEffect } from 'react'

import { resolveDetailsMode, resolveSections } from '../domain/details.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { ConfigFullResponse, ConfigMtimeResponse, ReloadMcpResponse } from '../gatewayTypes.js'
import { syncTuiLocale } from '../i18n/loader.js'
import { t } from '../i18n/runtime.js'
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
import { patchUiState } from './uiStore.js'

const STATUSBAR_ALIAS: Record<string, StatusBarMode> = {
  bottom: 'bottom',
  off: 'off',
  on: 'top',
  top: 'top'
}

export const normalizeStatusBar = (raw: unknown): StatusBarMode =>
  raw === false ? 'off' : typeof raw === 'string' ? (STATUSBAR_ALIAS[raw.trim().toLowerCase()] ?? 'top') : 'top'

// `display.status_bar.fields` — the SAME key the classic CLI bar honors
// (PR #98250). A non-empty list filters status-rule segments; missing/empty/
// malformed = null (user hasn't customized → show the default set). Unknown
// names pass through harmlessly — the renderer only tests membership.
export const normalizeStatusBarFields = (raw: unknown): null | ReadonlySet<string> => {
  if (!Array.isArray(raw) || raw.length === 0) {
    return null
  }

  const cleaned = raw.map(v => String(v).trim().toLowerCase()).filter(Boolean)

  return cleaned.length ? new Set(cleaned) : null
}

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
 * Extracted so the mtime-reload path can be exercised by the test
 * suite without a React runtime (Copilot round-12 review on #19835).
 * Both the initial hydration and the mtime poller use this shared
 * helper, so a regression in the fetch/apply plumbing now fails the
 * useConfigSync tests instead of only being visible at runtime. */
export async function hydrateFullConfig(
  gw: GatewayClient,
  setBell: (v: boolean) => void,
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void,
  setBellOnPrompt?: (v: boolean) => void
): Promise<ConfigFullResponse | null> {
  const cfg = await quietRpc<ConfigFullResponse>(gw, 'config.get', { key: 'full' })
  applyDisplay(cfg, setBell, setVoiceRecordKey, setBellOnPrompt)

  // Same fail-safe as the voice key: a null config (transient RPC failure)
  // keeps the last language rather than snapping back to English.
  if (cfg) {
    void syncTuiLocale(gw, cfg.config?.display?.language)
  }

  return cfg
}

export const applyDisplay = (
  cfg: ConfigFullResponse | null,
  setBell: (v: boolean) => void,
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void,
  setBellOnPrompt?: (v: boolean) => void
) => {
  const d = cfg?.config?.display ?? {}
  const approvals = cfg?.config?.approvals

  setBell(!!d.bell_on_complete)

  setBellOnPrompt?.(!!d.bell_on_prompt)

  applyConfiguredTuiTheme(d.tui_theme)

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
    battery: !!d.battery,
    busyInputMode: normalizeBusyInputMode(d.busy_input_mode),
    compact: !!d.tui_compact,
    // Fail safe: only YAML boolean false disables the prompt. A transient
    // config RPC failure (cfg=null) preserves the last known policy instead
    // of silently changing approval behavior until the next successful poll.
    ...(cfg ? { destructiveSlashConfirm: approvals?.destructive_slash_confirm !== false } : {}),
    detailsMode: resolveDetailsMode(d),
    detailsModeCommandOverride: false,
    focusView: !!d.focus_view,
    indicatorStyle: normalizeIndicatorStyle(d.tui_status_indicator),
    inlineDiffs: d.inline_diffs !== false,
    mouseTracking: normalizeMouseTracking(d),
    pasteCollapseLines: _pasteCollapseLinesFromConfig(cfg),
    pasteCollapseChars: _pasteCollapseCharsFromConfig(cfg),
    sections: resolveSections(d.sections),
    showReasoning: !!d.show_reasoning,
    statusBar: normalizeStatusBar(d.tui_statusbar),
    statusBarFields: normalizeStatusBarFields(d.status_bar?.fields),
    streaming: d.streaming !== false,
    // The SAME key that stamps [HH:MM] on classic-CLI labels (#41531) —
    // no separate TUI knob.
    timestamps: d.timestamps === true
  })
}

export function useConfigSync({
  gw,
  setBellOnComplete,
  setBellOnPrompt,
  setVoiceEnabled,
  setVoiceRecordKey,
  sid
}: UseConfigSyncOptions) {
  useEffect(() => {
    if (!sid) {
      return
    }

    setVoiceEnabled(process.env.HERMES_VOICE === '1')
    let disposed = false
    let running = false
    let dirty = false
    let forceHydrate = true
    let initialized = false
    let mtime = 0
    let timer: ReturnType<typeof setTimeout> | undefined
    const mcp: McpRevState = { accepted: '', inFlight: false }

    const refresh = async () => {
      if (disposed) return
      if (running) {
        dirty = true
        return
      }
      running = true
      clearTimeout(timer)
      let delay = MTIME_POLL_MS
      try {
        do {
          dirty = false
          const force = forceHydrate
          forceHydrate = false
          const r = await quietRpc<ConfigMtimeResponse>(gw, 'config.get', { key: 'mtime' })
          if (disposed) return
          if (!r) {
            forceHydrate ||= force
            break
          }
          const next = Number(r.mtime ?? 0)
          const rev = String(r.mcp_rev ?? '')
          const changed = initialized && next !== mtime
          if (!initialized) mcp.accepted = rev
          delay = r.change_events ? 60_000 : MTIME_POLL_MS

          if (force || !initialized || changed) {
            // Do not acknowledge a failed hydration or reset last-good display settings.
            const cfg = await quietRpc<ConfigFullResponse>(gw, 'config.get', { key: 'full' })
            if (disposed) return
            if (cfg) {
              applyDisplay(cfg, setBellOnComplete, setVoiceRecordKey, setBellOnPrompt)
              void syncTuiLocale(gw, cfg.config?.display?.language)
              mtime = next
            } else {
              forceHydrate = true
              delay = MTIME_POLL_MS
            }
          }
          if (initialized && rev) {
            await syncMcpReload(gw, sid, rev, mcp, () => {
              if (!disposed) turnController.pushActivity(t('status.mcpReloaded'))
            })
            if (mcp.accepted !== rev) delay = MTIME_POLL_MS
          } else if (initialized && changed && !rev) {
            const result = await quietRpc<ReloadMcpResponse>(gw, 'reload.mcp', { session_id: sid, confirm: true })
            if (!result || result.status !== 'reloaded') {
              mtime = Number.NaN
              delay = MTIME_POLL_MS
            }
          }
          initialized = true
        } while (dirty && !disposed)
      } finally {
        running = false
        if (!disposed)
          timer = setTimeout(() => {
            // The slow backstop also covers changes made before the watcher seeded.
            if (delay === 60_000) forceHydrate = true
            void refresh()
          }, delay)
      }
    }
    const onEvent = (event: { type: string }) => {
      if (event.type === 'config.changed' || event.type === 'gateway.ready') {
        forceHydrate = true
        void refresh()
      }
    }
    // Subscribe before the initial reads; events during either RPC queue another pass.
    gw.on('event', onEvent)
    void refresh()
    return () => {
      disposed = true
      clearTimeout(timer)
      gw.off('event', onEvent)
    }
  }, [gw, setBellOnComplete, setBellOnPrompt, setVoiceEnabled, setVoiceRecordKey, sid])
}

export interface UseConfigSyncOptions {
  gw: GatewayClient
  setBellOnComplete: (v: boolean) => void
  setBellOnPrompt?: (v: boolean) => void
  setVoiceEnabled: (v: boolean) => void
  setVoiceRecordKey?: (v: ParsedVoiceRecordKey) => void
  sid: null | string
}
