import type { UsageModelData } from '@hermes/shared/billing'
import type {
  ConnectionRequestPayload,
  GatewayEvent,
  GatewayEventName,
  InflightTurn,
  TranscriptMessage,
  Usage
} from '@hermes/shared/gateway-events'
import type { HermesSkin } from '@hermes/shared/skin'

import type { SessionInfo, SlashCategory } from './types.js'

/** The cross-surface skin contract (canonical shape in `@hermes/shared`).
 *  Includes the paired light_colors/dark_colors overlays from #20379. */
export type GatewaySkin = HermesSkin

/** Distributive form of the shared `GatewayEvent<K>` so `switch (ev.type)`
 *  narrows `ev.payload` per case (the generic-defaulted interface does not). */
export type AnyGatewayEvent = { [K in GatewayEventName]: GatewayEvent<K> }[GatewayEventName]

export interface GatewayCompletionItem {
  display: string
  /** Completion class, set by the gateway. `skill` covers skill commands and
   *  skill bundles — the only kind offered for an inline `/skill` reference. */
  kind?: string
  meta?: string
  text: string
}

// ── Commands / completion ────────────────────────────────────────────

export interface CommandsCatalogResponse {
  canon?: Record<string, string>
  categories?: SlashCategory[]
  pairs?: [string, string][]
  skill_count?: number
  sub?: Record<string, string[]>
  warning?: string
}

export interface CompletionResponse {
  items?: GatewayCompletionItem[]
  replace_from?: number
}

export interface SlashExecResponse {
  output?: string
  warning?: string
}

// ── Remote Spending (Phase 2b) ───────────────────────────────────────

// Wire shapes now live in @hermes/shared for reuse by TypeScript clients.
export type {
  BillingAutoReload,
  BillingBlock,
  BillingCardInfo,
  BillingChargeResponse,
  BillingChargeStatusResponse,
  BillingErrorPayload,
  BillingMonthlyCap,
  BillingMutationResponse,
  BillingStateResponse,
  SubscriptionPreviewResponse,
  SubscriptionStateResponse,
  SubscriptionTierOption,
  SubscriptionUpgradeResponse,
  UsageBarData,
  UsageModelData
} from '@hermes/shared/billing'

// ── Config ───────────────────────────────────────────────────────────

export interface ConfigDisplayConfig {
  battery?: boolean
  bell_on_complete?: boolean
  bell_on_prompt?: boolean
  busy_input_mode?: string
  details_mode?: string
  /** Focus view (/focus) — display-only reduced-output mode. */
  focus_view?: boolean
  inline_diffs?: boolean
  mouse_tracking?: boolean | null | number | string
  sections?: Record<string, string>
  show_cost?: boolean
  show_reasoning?: boolean
  /** CLI/TUI status-bar field visibility filter (shared with the classic
   *  CLI bar — see display.status_bar.fields in configuration docs).
   *  Raw YAML: callers must runtime-validate entries. */
  status_bar?: { fields?: unknown }
  streaming?: boolean
  thinking_mode?: string
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
  tui_mouse?: boolean | null | number | string
  // Forward-compat: backend may send styles this client doesn't know yet —
  // `normalizeIndicatorStyle` falls back to 'kaomoji' for those — but the
  // wire type is documented as `string` so consumers don't get a false
  // narrowing-and-autocomplete contract on a value that requires runtime
  // validation anyway.
  tui_status_indicator?: string
  tui_statusbar?: 'bottom' | 'off' | 'on' | 'top' | boolean
  /** Theme mode pin: 'light' / 'dark' beat background auto-detection; 'auto'
   *  (default) trusts the OSC-11 probe + env signals. */
  tui_theme?: string
}

export interface ConfigVoiceConfig {
  // Raw `yaml.safe_load()` values from config may be non-string if hand-edited.
  // Callers must normalize/validate at runtime.
  record_key?: unknown
  submit_mode?: unknown
}

export interface ConfigApprovalsConfig {
  // Raw config value: only the explicit boolean false disables the safety gate.
  destructive_slash_confirm?: unknown
}

export interface ConfigFullResponse {
  config?: {
    approvals?: ConfigApprovalsConfig
    display?: ConfigDisplayConfig
    voice?: ConfigVoiceConfig
    paste_collapse_threshold?: number
    paste_collapse_char_threshold?: number
  }
}

export interface ConfigMtimeResponse {
  /** Revision hash of MCP-relevant config sections; reload MCP only when it
   *  changes (cosmetic writes like /skin must not trigger reconnects). */
  mcp_rev?: string
  mtime?: number
}

export interface ConfigGetValueResponse {
  display?: string
  home?: string
  value?: string
}

export interface ConfigSetResponse {
  confirm_message?: string
  confirm_required?: boolean
  credential_warning?: string
  // A model pick made mid-turn is queued and applied at the next turn start,
  // not live yet — the handler says "next turn" instead of "model → X".
  deferred?: boolean
  history_reset?: boolean
  info?: SessionInfo
  value?: string
  warning?: string
}

export interface SetupStatusResponse {
  provider_configured?: boolean
}

export interface SystemBatteryResponse {
  available?: boolean
  category?: string
  percent?: null | number
  plugged?: null | boolean
}

// ── Session lifecycle ────────────────────────────────────────────────

export interface SessionCreateResponse {
  info?: SessionInfo & { config_warning?: string; credential_warning?: string }
  session_id: string
  // Durable id (state.db row) — what session.resume takes; `session_id` is the
  // process-local runtime handle.
  stored_session_id?: string
}

export type LiveSessionStatus = 'idle' | 'starting' | 'waiting' | 'working'

export interface SessionActiveItem {
  current?: boolean
  id: string
  last_active?: number
  message_count?: number
  model?: string
  preview?: string
  session_key?: string
  started_at?: number
  status: LiveSessionStatus
  title?: string
}

export interface SessionActiveListResponse {
  sessions?: SessionActiveItem[]
}

export interface SessionActivateResponse {
  inflight?: null | InflightTurn
  info?: SessionInfo
  message_count?: number
  messages: TranscriptMessage[]
  pending_connection?: ConnectionRequestPayload | null
  running?: boolean
  session_id: string
  session_key?: string
  started_at?: number
  status?: LiveSessionStatus
}

export interface SessionDeleteResponse {
  deleted: string
}

export interface SessionMostRecentResponse {
  session_id?: null | string
  source?: string
  started_at?: number
  title?: string
}

export interface SessionTitleResponse {
  pending?: boolean
  session_key?: string
  title?: string
}

export interface SessionSaveResponse {
  file?: string
}

export interface SessionUndoResponse {
  removed?: number
}

export interface SessionUsageResponse {
  active_subagents?: number
  avg_latency_s?: number
  avg_tps?: number
  cache_hit_pct?: number
  cache_read?: number
  cache_write?: number
  calls?: number
  compressions?: number
  context_max?: number
  context_percent?: number
  context_estimated?: boolean
  context_source?: string
  context_used?: number
  cost_status?: 'estimated' | 'exact'
  cost_usd?: number
  credits_lines?: string[]
  input?: number
  model?: string
  output?: number
  total?: number
  // Shared dollar usage model (two-bar view) so /usage renders the same bars
  // as /subscription. Dollars only — never "credits".
  usage?: UsageModelData
}

export interface SessionStatusResponse {
  output?: string
}

export interface SessionCompressResponse {
  after_messages?: number
  after_tokens?: number
  before_messages?: number
  before_tokens?: number
  info?: SessionInfo
  messages?: TranscriptMessage[]
  removed?: number
  summary?: {
    headline?: string
    noop?: boolean
    note?: null | string
    token_line?: string
  }
  usage?: Usage
}

export interface SessionBranchResponse {
  session_id?: string
  title?: string
}

export interface SessionCloseResponse {
  closed?: boolean
  ok?: boolean
}

export interface SessionInterruptResponse {
  ok?: boolean
}

export interface SessionSteerResponse {
  status?: 'queued' | 'rejected'
  text?: string
}

// ── Prompt / submission ──────────────────────────────────────────────

export interface PromptSubmitResponse {
  ok?: boolean
  /** Set when the submitted text was a bare voice stop phrase consumed
   *  server-side to end the voice chat instead of starting a turn. */
  voice_stopped?: boolean
}

export interface BackgroundStartResponse {
  task_id?: string
}

/** `clarify.lock` — one batch-clarify answer locked; `expired` when the request already ended. */
export interface ClarifyLockResponse {
  remaining?: string[]
  status: 'expired' | 'ok'
}

// ── Shell / clipboard / input ────────────────────────────────────────

export interface ShellExecResponse {
  code: number
  stderr?: string
  stdout?: string
}

export interface ClipboardPasteResponse {
  attached?: boolean
  count?: number
  height?: number
  message?: string
  token_estimate?: number
  width?: number
}

export interface InputDetectDropResponse {
  height?: number
  is_image?: boolean
  matched?: boolean
  name?: string
  text?: string
  token_estimate?: number
  width?: number
}

export interface TerminalResizeResponse {
  ok?: boolean
}

// ── Image attach ─────────────────────────────────────────────────────

export interface ImageAttachResponse {
  height?: number
  name?: string
  remainder?: string
  token_estimate?: number
  width?: number
}

// ── Voice ────────────────────────────────────────────────────────────

export interface VoiceToggleResponse {
  audio_available?: boolean
  available?: boolean
  details?: string
  enabled?: boolean
  record_key?: string
  stop_hint?: string
  stt_available?: boolean
  tts?: boolean
}

export interface VoiceRecordResponse {
  status?: 'busy' | 'recording' | 'stopped'
  text?: string
}

// ── Wake word ────────────────────────────────────────────────────────

export interface WakeStartResponse {
  enabled_persisted?: boolean
  hint?: string
  owner_surface?: null | string
  phrase?: string
  provider?: string
  reason?: string
  started?: boolean
}

export interface WakeStopResponse {
  disabled_persisted?: boolean
  reason?: null | string
  stopped?: boolean
}

export interface WakeStatusResponse {
  /** Armed but the mic delivers only silence (macOS backend-permission gap). */
  audio_silent?: boolean
  available?: boolean
  /** Config truth (wake_word.enabled). */
  enabled?: boolean
  hint?: string
  listening?: boolean
  owned_by_caller?: boolean
  owner_surface?: null | string
  phrase?: string
  provider?: string
}

// ── Tools (TS keeps configure since it resets local history) ─────────

export interface ToolsConfigureResponse {
  changed?: string[]
  enabled_toolsets?: string[]
  info?: SessionInfo
  missing_servers?: string[]
  reset?: boolean
  unknown?: string[]
}

// ── MCP ──────────────────────────────────────────────────────────────

export interface ReloadMcpResponse {
  status?: string
  message?: string
  /** The mcp_rev the server actually loaded (re-hashed after discovery).
   *  The client records THIS as its accepted revision, not the one it
   *  requested — a reload that raced a config edit reports the newer rev. */
  loaded_rev?: string
}

export interface ReloadEnvResponse {
  updated?: number
}

export interface ProcessStopResponse {
  killed?: number
}

export interface BrowserManageResponse {
  connected?: boolean
  messages?: string[]
  url?: string
}

export interface RollbackCheckpoint {
  hash: string
  message?: string
  timestamp?: string
}

export interface RollbackListResponse {
  checkpoints?: RollbackCheckpoint[]
  enabled?: boolean
}

export interface RollbackDiffResponse {
  diff?: string
  rendered?: string
  stat?: string
}

export interface RollbackRestoreResponse {
  error?: string
  history_removed?: number
  message?: string
  reason?: string
  restored_to?: string
  success?: boolean
}

// ── Delegation control RPCs ──────────────────────────────────────────

export interface DelegationStatusResponse {
  active?: {
    depth?: number
    goal?: string
    model?: null | string
    parent_id?: null | string
    started_at?: number
    status?: string
    subagent_id?: string
    tool_count?: number
  }[]
  max_concurrent_children?: number
  max_spawn_depth?: number
  paused?: boolean
}

export interface DelegationPauseResponse {
  paused?: boolean
}

export interface AsyncDelegationRecord {
  delegation_id: string
  goal?: string | null
  role?: string | null
  model?: string | null
  status?: string | null
  dispatched_at?: number | null
  completed_at?: number | null
  subagent_ids?: string[]
}

export interface SubagentListResponse {
  subagents: {
    subagent_id: string
    parent_id?: string | null
    delegation_id?: string | null
    depth?: number | null
    goal?: string | null
    model?: string | null
    started_at?: number | null
    status?: string | null
    tool_count?: number | null
    last_tool?: string | null
  }[]
  delegations: AsyncDelegationRecord[]
}

export interface SubagentInterruptResponse {
  found?: boolean
  subagent_id?: string
}

// ── Spawn-tree snapshots ─────────────────────────────────────────────

export interface SpawnTreeListEntry {
  count: number
  finished_at?: number
  label?: string
  path: string
  session_id?: string
  started_at?: number | null
}

export interface SpawnTreeListResponse {
  entries?: SpawnTreeListEntry[]
}

export interface SpawnTreeLoadResponse {
  finished_at?: number
  label?: string
  session_id?: string
  started_at?: null | number
  subagents?: unknown[]
}
