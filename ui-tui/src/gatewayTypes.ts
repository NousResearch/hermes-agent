import type {
  ConfigGetResult,
  ConfigSetResult,
  JsonValue,
  SessionLiveInfo,
  SkinPayload
} from '@hermes/shared/gateway-events'

/** The resolved skin as the gateway sends it (`gateway.ready`, `skin.changed`).
 *  Includes the paired light_colors/dark_colors overlays from #20379. */
export type GatewaySkin = SkinPayload

// ── Config ───────────────────────────────────────────────────────────

/** `display.*` in the raw config.yaml tree. Hand-written on purpose: the wire
 *  contract types `config.get {key:'full'}` as opaque JSON, and every value here
 *  may be hand-edited YAML, so each read normalizes. */
export interface ConfigDisplayConfig {
  battery?: boolean
  bell_on_complete?: boolean
  bell_on_prompt?: boolean
  busy_input_mode?: JsonValue
  details_mode?: JsonValue
  /** Focus view (/focus) — display-only reduced-output mode. */
  focus_view?: boolean
  inline_diffs?: boolean
  mouse_tracking?: JsonValue
  sections?: JsonValue
  show_cost?: boolean
  show_reasoning?: boolean
  /** CLI/TUI status-bar field visibility filter (shared with the classic
   *  CLI bar — see display.status_bar.fields in configuration docs). */
  status_bar?: { fields?: JsonValue }
  streaming?: boolean
  thinking_mode?: JsonValue
  /** Show [HH:MM] timestamps on transcript rows — same key the classic CLI
   *  honors on its user/assistant labels (#41531). */
  timestamps?: boolean
  /**
   * Nudge the user toward the /agents spawn-tree dashboard the first time a
   * turn starts delegating, via a one-time transient activity hint.  Opens
   * nothing — just advertises the command.  Default true.
   */
  tui_agents_nudge?: boolean
  tui_auto_resume_recent?: boolean
  tui_compact?: boolean
  /** Legacy alias for display.mouse_tracking. */
  tui_mouse?: JsonValue
  // Forward-compat: the backend may send styles this client doesn't know yet —
  // `normalizeIndicatorStyle` falls back to 'kaomoji' for those.
  tui_status_indicator?: JsonValue
  tui_statusbar?: JsonValue
  /** Theme mode pin: 'light' / 'dark' beat background auto-detection; 'auto'
   *  (default) trusts the OSC-11 probe + env signals. */
  tui_theme?: string
}

export interface ConfigVoiceConfig {
  record_key?: JsonValue
  submit_mode?: JsonValue
}

export interface ConfigApprovalsConfig {
  /** Only the explicit boolean false disables the safety gate. */
  destructive_slash_confirm?: JsonValue
}

/** The slices of the raw config.yaml tree the TUI reads. */
export interface HermesConfigTree {
  approvals?: ConfigApprovalsConfig
  display?: ConfigDisplayConfig
  voice?: ConfigVoiceConfig
  paste_collapse_threshold?: JsonValue
  paste_collapse_char_threshold?: JsonValue
}

/** The one place `config.get`'s opaque `config` JSON becomes the tree above. */
export const hermesConfigTree = (result: ConfigGetResult | null): HermesConfigTree | null =>
  // SAFETY: the wire field is the raw YAML tree as JSON; every field read off it is optional and normalized.
  (result?.config as HermesConfigTree | null) ?? null

/** `config.set` echoes the normalized value as JSON; every TUI surface shows it as text. */
export const configValueText = (value: JsonValue): string => {
  if (value === null) {
    return ''
  }

  const json = JSON.stringify(value)

  // Only a JSON string starts with a quote; it shows raw, every other kind shows as its JSON text.
  return json.startsWith('"') ? String(JSON.parse(json)) : json
}

/** `config.set personality` echoes the refreshed session snapshot as opaque JSON. */
export const configSetSessionInfo = (result: ConfigSetResult): SessionLiveInfo | null =>
  // SAFETY: methods_config_set._set_personality writes the session snapshot into this field.
  (result.info as SessionLiveInfo | null) ?? null
