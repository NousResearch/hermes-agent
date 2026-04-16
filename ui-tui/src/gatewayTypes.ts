import type { SessionInfo, SlashCategory, Usage } from './types.js'

export interface GatewaySkin {
  banner_hero?: string
  banner_logo?: string
  branding?: Record<string, string>
  colors?: Record<string, string>
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

export type CommandDispatchResponse =
  | { output?: string; type: 'exec' | 'plugin' }
  | { target: string; type: 'alias' }
  | { message?: string; name: string; type: 'skill' }

// ── Config ───────────────────────────────────────────────────────────

export interface ConfigDisplayConfig {
  bell_on_complete?: boolean
  details_mode?: string
  thinking_mode?: string
  tui_compact?: boolean
  tui_statusbar?: boolean
}

export interface ConfigFullResponse {
  config?: { display?: ConfigDisplayConfig }
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
  credential_warning?: string
  history_reset?: boolean
  info?: SessionInfo
  value?: string
  warning?: string
}

// ── Session lifecycle ────────────────────────────────────────────────

export interface SessionCreateResponse {
  info?: SessionInfo & { credential_warning?: string }
  session_id: string
}

export interface SessionResumeResponse {
  info?: SessionInfo
  message_count?: number
  messages: GatewayTranscriptMessage[]
  resumed?: string
  session_id: string
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

export interface SessionUndoResponse {
  removed?: number
}

export interface SessionHistoryResponse {
  count?: number
  messages?: GatewayTranscriptMessage[]
}

export interface SessionCompressResponse {
  info?: SessionInfo
  messages?: GatewayTranscriptMessage[]
  removed?: number
  usage?: Usage
}

export interface SessionBranchResponse {
  session_id?: string
  title?: string
}

export interface SessionTitleResponse {
  title?: string
}

export interface SessionSaveResponse {
  file?: string
}

export interface SessionUsageResponse {
  cache_read?: number
  cache_write?: number
  calls?: number
  compressions?: number
  context_max?: number
  context_percent?: number
  context_used?: number
  cost_status?: 'estimated' | 'exact'
  cost_usd?: number
  input?: number
  model?: string
  output?: number
  total?: number
}

export interface SessionCloseResponse {
  ok?: boolean
}

export interface SessionInterruptResponse {
  ok?: boolean
}

// ── Prompt / submission ──────────────────────────────────────────────

export interface PromptSubmitResponse {
  ok?: boolean
}

export interface BackgroundStartResponse {
  task_id?: string
}

export interface BtwStartResponse {
  ok?: boolean
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
  enabled?: boolean
}

export interface VoiceRecordResponse {
  text?: string
}

// ── Tools / toolsets ─────────────────────────────────────────────────

export interface ToolsetDetails {
  description: string
  enabled: boolean
  name: string
  tool_count: number
  tools: string[]
}

export interface ToolsListResponse {
  toolsets?: ToolsetDetails[]
}

export interface ToolSummary {
  description: string
  name: string
}

export interface ToolsShowSection {
  name: string
  tools: ToolSummary[]
}

export interface ToolsShowResponse {
  sections?: ToolsShowSection[]
  total?: number
}

export interface ToolsConfigureResponse {
  changed?: string[]
  enabled_toolsets?: string[]
  info?: SessionInfo
  missing_servers?: string[]
  reset?: boolean
  unknown?: string[]
}

export interface ToolsetsListResponse {
  toolsets?: {
    description: string
    enabled: boolean
    name: string
    tool_count: number
  }[]
}

// ── Ops: rollback / browser / plugins / skills / agents / cron ───────

export interface RollbackCheckpoint {
  hash?: string
  message?: string
}

export interface RollbackListResponse {
  checkpoints?: RollbackCheckpoint[]
}

export interface RollbackActionResponse {
  diff?: string
  message?: string
  rendered?: string
}

export interface BrowserManageResponse {
  connected?: boolean
  url?: string
}

export interface PluginInfo {
  enabled?: boolean
  name?: string
  version?: string
}

export interface PluginsListResponse {
  plugins?: PluginInfo[]
}

export interface SkillsListResponse {
  skills?: Record<string, string[]>
}

export interface SkillsBrowseItem {
  description?: string
  name?: string
}

export interface SkillsBrowseResponse {
  items?: SkillsBrowseItem[]
  page?: number
  total?: number
  total_pages?: number
}

export interface AgentProcess {
  command?: string
  session_id: string
  status?: 'finished' | 'running'
}

export interface AgentsListResponse {
  processes?: AgentProcess[]
}

export interface CronJob {
  job_id?: string
  name?: string
  schedule?: string
  state?: string
}

export interface CronListResponse {
  jobs?: CronJob[]
}

export interface ConfigShowSection {
  rows?: [string, string][]
  title?: string
}

export interface ConfigShowResponse {
  sections?: ConfigShowSection[]
}

// ── Insights / MCP ───────────────────────────────────────────────────

export interface InsightsResponse {
  days?: number
  messages?: number
  sessions?: number
}

export interface ReloadMcpResponse {
  ok?: boolean
}

export interface ModelOptionProvider {
  is_current?: boolean
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

// ── Subagent events ──────────────────────────────────────────────────

export interface SubagentEventPayload {
  duration_seconds?: number
  goal: string
  status?: 'completed' | 'failed' | 'interrupted' | 'running'
  summary?: string
  task_count?: number
  task_index: number
  text?: string
  tool_name?: string
  tool_preview?: string
}

export type GatewayEvent =
  | { payload?: { skin?: GatewaySkin }; session_id?: string; type: 'gateway.ready' }
  | { payload?: GatewaySkin; session_id?: string; type: 'skin.changed' }
  | { payload: SessionInfo; session_id?: string; type: 'session.info' }
  | { payload?: { text?: string }; session_id?: string; type: 'thinking.delta' }
  | { payload?: undefined; session_id?: string; type: 'message.start' }
  | { payload?: { kind?: string; text?: string }; session_id?: string; type: 'status.update' }
  | { payload: { line: string }; session_id?: string; type: 'gateway.stderr' }
  | { payload?: { cwd?: string; python?: string }; session_id?: string; type: 'gateway.start_timeout' }
  | { payload?: { preview?: string }; session_id?: string; type: 'gateway.protocol_error' }
  | { payload?: { text?: string }; session_id?: string; type: 'reasoning.delta' | 'reasoning.available' }
  | { payload: { name?: string; preview?: string }; session_id?: string; type: 'tool.progress' }
  | { payload: { name?: string }; session_id?: string; type: 'tool.generating' }
  | { payload: { context?: string; name?: string; tool_id: string }; session_id?: string; type: 'tool.start' }
  | {
      payload: { error?: string; inline_diff?: string; name?: string; summary?: string; tool_id: string }
      session_id?: string
      type: 'tool.complete'
    }
  | {
      payload: { choices: string[] | null; question: string; request_id: string }
      session_id?: string
      type: 'clarify.request'
    }
  | { payload: { command: string; description: string }; session_id?: string; type: 'approval.request' }
  | { payload: { request_id: string }; session_id?: string; type: 'sudo.request' }
  | { payload: { env_var: string; prompt: string; request_id: string }; session_id?: string; type: 'secret.request' }
  | { payload: { task_id: string; text: string }; session_id?: string; type: 'background.complete' }
  | { payload: { text: string }; session_id?: string; type: 'btw.complete' }
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
