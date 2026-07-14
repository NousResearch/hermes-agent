import type { SessionInfo, SlashCategory, SubagentStatus, Usage } from './types.js'

export interface GatewaySkin {
  banner_hero?: string
  banner_logo?: string
  branding?: Record<string, string>
  colors?: Record<string, string>
  help_header?: string
  tool_prefix?: string
}

export interface GatewayCompletionItem {
  display: string
  meta?: string
  text: string
}

export interface GatewayTranscriptMessage {
  context?: string
  name?: string
  role: 'assistant' | 'system' | 'tool' | 'user'
  text?: string
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

// ── Credits / top-up ─────────────────────────────────────────────────

export interface CreditsViewResponse {
  balance_lines: string[]
  depleted: boolean
  identity_line: string | null
  logged_in: boolean
  topup_url: string | null
}

// ── Terminal billing (Phase 2b) ──────────────────────────────────────

export interface BillingCardInfo {
  brand: string
  last4: string
  masked: string
}

export interface BillingMonthlyCap {
  is_default_ceiling: boolean
  limit_display: string
  limit_usd: string | null
  spent_display: string
  spent_this_month_usd: string | null
}

export interface BillingAutoReload {
  enabled: boolean
  reload_to_display: string
  reload_to_usd: string | null
  threshold_display: string
  threshold_usd: string | null
}

export interface BillingStateResponse {
  auto_reload: BillingAutoReload | null
  balance_display: string
  balance_usd: string | null
  can_charge: boolean
  card: BillingCardInfo | null
  charge_presets: string[]
  charge_presets_display: string[]
  cli_billing_enabled: boolean
  error?: string | null
  is_admin: boolean
  logged_in: boolean
  max_usd: string | null
  min_usd: string | null
  monthly_cap: BillingMonthlyCap | null
  ok: boolean
  org_name: string | null
  portal_url: string | null
  role: string | null
}

/**
 * Raw error payload echoed from the server (`_serialize_billing_error`). Carries
 * the extra fields a few error codes attach — notably `remainingUsd` on
 * `monthly_cap_exceeded` — so the client can render the same detail the CLI does.
 */
export interface BillingErrorPayload {
  isDefaultCeiling?: boolean
  remainingUsd?: string
}

export interface BillingChargeResponse {
  charge_id?: string
  error?: string
  idempotency_key?: string
  message?: string
  ok: boolean
  payload?: BillingErrorPayload
  portal_url?: string | null
  retry_after?: number | null
}

export interface BillingChargeStatusResponse {
  amount_usd?: string | null
  error?: string
  message?: string
  ok: boolean
  payload?: BillingErrorPayload
  portal_url?: string | null
  reason?: string | null
  retry_after?: number | null
  settled_at?: string | null
  status?: string
}

export interface BillingMutationResponse {
  error?: string
  granted?: boolean
  message?: string
  ok: boolean
  payload?: BillingErrorPayload
  portal_url?: string | null
  retry_after?: number | null
}

export type CommandDispatchResponse =
  | { output?: string; type: 'exec' | 'plugin' }
  | { target: string; type: 'alias' }
  | { message?: string; name: string; type: 'skill' }
  | { message: string; notice?: string; type: 'send' }
  | { message: string; notice?: string; type: 'prefill' }

// ── Config ───────────────────────────────────────────────────────────

export interface ConfigDisplayConfig {
  bell_on_complete?: boolean
  busy_input_mode?: string
  details_mode?: string
  inline_diffs?: boolean
  mouse_tracking?: boolean | null | number | string
  sections?: Record<string, string>
  show_cost?: boolean
  show_reasoning?: boolean
  streaming?: boolean
  thinking_mode?: string
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
}

export interface ConfigVoiceConfig {
  // Raw `yaml.safe_load()` value from config; may be non-string if hand-edited.
  // Callers must normalize/validate at runtime (parseVoiceRecordKey()).
  record_key?: unknown
}

export interface ConfigFullResponse {
  config?: {
    display?: ConfigDisplayConfig
    voice?: ConfigVoiceConfig
    paste_collapse_threshold?: number
    paste_collapse_char_threshold?: number
  }
}

export interface ConfigMtimeResponse {
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
  history_reset?: boolean
  info?: SessionInfo
  value?: string
  warning?: string
}

export interface SetupStatusResponse {
  provider_configured?: boolean
}

// ── Session lifecycle ────────────────────────────────────────────────

export interface SessionCreateResponse {
  info?: SessionInfo & { config_warning?: string; credential_warning?: string }
  session_id: string
}

export interface SessionResumeResponse {
  inflight?: null | SessionInflightTurn
  info?: SessionInfo
  message_count?: number
  messages: GatewayTranscriptMessage[]
  resumed?: string
  running?: boolean
  session_id: string
  started_at?: number
  status?: LiveSessionStatus
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

export interface SessionInflightTurn {
  assistant?: string
  streaming?: boolean
  user?: string
}

export interface SessionActivateResponse {
  inflight?: null | SessionInflightTurn
  info?: SessionInfo
  message_count?: number
  messages: GatewayTranscriptMessage[]
  running?: boolean
  session_id: string
  session_key?: string
  started_at?: number
  status?: LiveSessionStatus
}

export interface SessionListItem {
  id: string
  message_count: number
  preview: string
  source?: string
  started_at: number
  title: string
}

export interface SessionListResponse {
  sessions?: SessionListItem[]
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
  account_label?: string
  account_label_short?: string
  account_label_tiny?: string
  account_level?: 'error' | 'ok' | 'warn'
  active_subagents?: number
  cache_read?: number
  cache_write?: number
  calls?: number
  compressions?: number
  context_max?: number
  context_percent?: number
  context_used?: number
  cost_status?: 'estimated' | 'exact'
  cost_usd?: number
  credits_lines?: string[]
  input?: number
  model?: string
  output?: number
  total?: number
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
  messages?: GatewayTranscriptMessage[]
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
}

export interface BackgroundStartResponse {
  task_id?: string
}

export interface ClarifyRespondResponse {
  ok?: boolean
}

export interface ApprovalRespondResponse {
  ok?: boolean
}

export interface SudoRespondResponse {
  ok?: boolean
}

export interface SecretRespondResponse {
  ok?: boolean
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
  stt_available?: boolean
  tts?: boolean
}

export interface VoiceRecordResponse {
  status?: 'busy' | 'recording' | 'stopped'
  text?: string
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

// ── Model picker ─────────────────────────────────────────────────────

export interface ModelOptionProvider {
  auth_type?: string
  authenticated?: boolean
  is_current?: boolean
  key_env?: string
  models?: string[]
  name: string
  slug: string
  total_models?: number
  warning?: string
}

export interface ModelOptionsResponse {
  model?: string
  provider?: string
  providers?: ModelOptionProvider[]
}

// ── MCP ──────────────────────────────────────────────────────────────

export interface ReloadMcpResponse {
  status?: string
  message?: string
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

// ── Subagent events ──────────────────────────────────────────────────

export interface SubagentEventPayload {
  api_calls?: number
  cost_usd?: number
  depth?: number
  duration_seconds?: number
  files_read?: string[]
  files_written?: string[]
  goal: string
  input_tokens?: number
  iteration?: number
  model?: string
  output_tail?: { is_error?: boolean; preview?: string; tool?: string }[]
  output_tokens?: number
  parent_id?: null | string
  reasoning_tokens?: number
  status?: SubagentStatus
  subagent_id?: string
  summary?: string
  task_count?: number
  task_index: number
  text?: string
  tool_count?: number
  tool_name?: string
  tool_preview?: string
  toolsets?: string[]
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

export type GatewayEvent =
  | { payload?: { skin?: GatewaySkin }; session_id?: string; type: 'gateway.ready' }
  | { payload?: GatewaySkin; session_id?: string; type: 'skin.changed' }
  | { payload: SessionInfo; session_id?: string; type: 'session.info' }
  | { payload?: { text?: string }; session_id?: string; type: 'thinking.delta' }
  | { payload?: { kind?: string }; session_id?: string; type: 'reaction' }
  | { payload?: undefined; session_id?: string; type: 'message.start' }
  | { payload?: { kind?: string; text?: string }; session_id?: string; type: 'status.update' }
  | {
      payload?: {
        id?: string
        key?: string
        kind?: 'sticky' | 'ttl'
        level?: 'error' | 'info' | 'success' | 'warn'
        text?: string
        ttl_ms?: null | number
      }
      session_id?: string
      type: 'notification.show'
    }
  | { payload?: { key?: string }; session_id?: string; type: 'notification.clear' }
  | {
      payload: { user_code?: string; verification_url: string }
      session_id?: string
      type: 'billing.step_up.verification'
    }
  | { payload?: { state?: 'idle' | 'listening' | 'transcribing' }; session_id?: string; type: 'voice.status' }
  | { payload?: { no_speech_limit?: boolean; text?: string }; session_id?: string; type: 'voice.transcript' }
  | { payload?: { reason?: string }; session_id?: string; type: 'dashboard.new_session_requested' }
  | { payload: { line: string }; session_id?: string; type: 'gateway.stderr' }
  | {
      payload?: { level?: 'info' | 'warn' | 'error'; message?: string }
      session_id?: string
      type: 'browser.progress'
    }
  | {
      payload?: { cwd?: string; python?: string; stderr_tail?: string }
      session_id?: string
      type: 'gateway.start_timeout'
    }
  | { payload?: { preview?: string }; session_id?: string; type: 'gateway.protocol_error' }
  | {
      payload?: { text?: string; verbose?: boolean }
      session_id?: string
      type: 'reasoning.delta' | 'reasoning.available'
    }
  | {
      payload: { count?: number; index?: number; label?: string; text?: string }
      session_id?: string
      type: 'moa.reference'
    }
  | { payload?: { aggregator?: string }; session_id?: string; type: 'moa.aggregating' }
  | { payload: { name?: string; preview?: string }; session_id?: string; type: 'tool.progress' }
  | { payload: { name?: string }; session_id?: string; type: 'tool.generating' }
  | {
      payload: { args_text?: string; context?: string; name?: string; tool_id: string; todos?: unknown[] }
      session_id?: string
      type: 'tool.start'
    }
  | {
      payload: {
        duration_s?: number
        error?: string
        inline_diff?: string
        name?: string
        result_text?: string
        summary?: string
        tool_id: string
        todos?: unknown[]
      }
      session_id?: string
      type: 'tool.complete'
    }
  | {
      payload: { choices: string[] | null; question: string; request_id: string }
      session_id?: string
      type: 'clarify.request'
    }
  | {
      payload: { allow_permanent?: boolean; choices?: string[]; command: string; description: string; smart_denied?: boolean }
      session_id?: string
      type: 'approval.request'
    }
  | { payload: { request_id: string }; session_id?: string; type: 'sudo.request' }
  | { payload: { env_var: string; prompt: string; request_id: string }; session_id?: string; type: 'secret.request' }
  | { payload: { request_id: string }; session_id?: string; type: 'secret.expire' | 'sudo.expire' }
  | { payload: { task_id: string; text: string }; session_id?: string; type: 'background.complete' }
  | { payload?: { text?: string }; session_id?: string; type: 'review.summary' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.spawn_requested' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.start' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.thinking' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.tool' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.progress' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.complete' }
  | { payload: { rendered?: string; text?: string }; session_id?: string; type: 'message.delta' }
  | {
      payload?: { reasoning?: string; rendered?: string; text?: string; usage?: Usage }
      session_id?: string
      type: 'message.complete'
    }
  | { payload?: { message?: string }; session_id?: string; type: 'error' }
