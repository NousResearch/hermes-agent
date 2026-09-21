import type { ProjectInfo, SessionLiveInfo, SubagentStatus } from '@hermes/shared/gateway-events'

export interface ActiveTool {
  context?: string
  id: string
  name: string
  verboseArgs?: string
  startedAt?: number
}

export interface TodoItem {
  content: string
  id: string
  /** Optional id of another item — renders this as a nested subtask. */
  parent?: string
  status: 'cancelled' | 'completed' | 'in_progress' | 'pending'
}

export interface ActivityItem {
  id: number
  text: string
  tone: 'error' | 'info' | 'warn'
}

export interface SubagentProgress {
  apiCalls?: number
  costUsd?: number
  /** Batch (delegation) id — tags `[n/N]` rows so concurrent/nested fan-outs
   *  are distinguishable. Absent on older gateways. */
  delegationId?: string
  depth: number
  durationSeconds?: number
  filesRead?: string[]
  filesWritten?: string[]
  goal: string
  id: string
  index: number
  inputTokens?: number
  iteration?: number
  model?: string
  notes: string[]
  outputTail?: SubagentOutputEntry[]
  outputTokens?: number
  parentId: null | string
  reasoningTokens?: number
  startedAt?: number
  status: SubagentStatus
  summary?: string
  taskCount: number
  thinking: string[]
  toolCount: number
  tools: string[]
  toolsets?: string[]
}

export interface SubagentOutputEntry {
  isError: boolean
  preview: string
  tool: string
}

export interface SubagentNode {
  aggregate: SubagentAggregate
  children: SubagentNode[]
  item: SubagentProgress
}

export interface SubagentAggregate {
  activeCount: number
  costUsd: number
  descendantCount: number
  filesTouched: number
  hotness: number
  inputTokens: number
  maxDepthFromHere: number
  outputTokens: number
  totalDuration: number
  totalTools: number
}

export interface DelegationStatus {
  active: {
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
  paused: boolean
}

export interface ApprovalReq {
  // false when the backend won't honor a permanent allow (tirith warning) → hide "Always allow".
  allowPermanent?: boolean
  choices?: string[]
  command: string
  description: string
  /** Server→client request id; the answer is the response frame for it. */
  requestId: string
  smartDenied?: boolean
}

export interface ConfirmReq {
  cancelLabel?: string
  confirmLabel?: string
  danger?: boolean
  detail?: string
  onConfirm: () => void
  title: string
}

export interface ClarifyBatchQuestion {
  choices: string[] | null
  header?: string
  multiSelect?: boolean
  options?: AskUserQuestionOption[]
  qid: string
  question: string
}

export interface ClarifyReq {
  choices: string[] | null
  /** Absolute Unix seconds when this prompt expires; absent means unlimited. */
  expiresAt?: number
  question: string
  requestId: string
  /** Batch (multi-question) clarify: present instead of question/choices. */
  questions?: ClarifyBatchQuestion[]
  /** Answers already locked server-side (qid → answer): seeded from the
   *  reconnect replay, updated as the user locks each question. */
  answers?: Record<string, string>
}

// ── KENSEI CUSTOM: AskUserQuestionsTool overlay types ────────────────
// Per spec (2026-06-04): the new mode system (plan / UltraPlan / recon)
// uses a multi-question batched prompt instead of single ClarifyPrompt.
// Mirrors Claude Code's AskUserQuestion tool.  Must survive upstream merges.
export interface AskUserQuestionOption {
  description?: string
  label: string
  recommended?: boolean
}

export interface AskUserQuestion {
  header?: string
  multiSelect?: boolean
  options: AskUserQuestionOption[]
  question: string
}

export interface AskUserQuestionsReq {
  questions: AskUserQuestion[]
  requestId: string
}

export interface Msg {
  info?: SessionInfo
  kind?: 'diff' | 'event' | 'intro' | 'panel' | 'slash' | 'trail'
  panelData?: PanelData
  role: Role
  text: string
  // Unix seconds the message was authored (persisted transcript timestamp on
  // rehydrate, wall clock at append time for live rows). Rendered as a dim
  // [HH:MM] label when `display.timestamps` is on (#41531).
  createdAt?: number
  thinking?: string
  // MoA reference-model output stored in `thinking` (see turnController's
  // recordMoaReference): unlike ordinary model reasoning, this is the
  // user-facing mixture-of-agents process the user opted into, so it stays
  // visible even when `display.sections.thinking` is hidden.
  isMoaReference?: boolean
  // True only while this trail segment's reasoning is being streamed live by
  // the current turn (see turnController's syncReasoningSegment). Sealed
  // reasoning segments from earlier in the turn carry no flag, so the TUI can
  // tell "the reasoning happening right now" apart from finished blocks.
  isLiveReasoning?: boolean
  thinkingTokens?: number
  toolTokens?: number
  tools?: string[]
  todos?: TodoItem[]
  todoIncomplete?: boolean
  todoCollapsedByDefault?: boolean
}

export type Role = 'assistant' | 'system' | 'tool' | 'user'
export type DetailsMode = 'hidden' | 'collapsed' | 'expanded'
export type ThinkingMode = 'collapsed' | 'truncated' | 'full'

// Per-section overrides for the agent details accordion.  Resolution order
// at lookup time is: explicit `display.sections.<name>` → built-in
// SECTION_DEFAULTS → global `details_mode`.  Today the built-in defaults
// expand `thinking`/`tools` and hide `activity`; `subagents` falls through
// to the global mode.  Any explicit value still wins for that one section.
export type SectionName = 'thinking' | 'tools' | 'subagents' | 'activity'
export type SectionVisibility = Partial<Record<SectionName, DetailsMode>>

export interface McpServerStatus {
  connected: boolean
  disabled?: boolean
  status?: 'configured' | 'connecting' | 'connected' | 'disabled' | 'failed' | 'lazy'
  name: string
  tools: number
  transport: string
}

/** The gateway's `session.info` / resume `info` block — generated from `tui_gateway/contracts`. */
// KENSEI CUSTOM: agent_mode is fork-only (agent-modes skill); intersect onto the generated shape.
export type SessionInfo = SessionLiveInfo & { agent_mode?: string }
export type { ProjectInfo }

export interface SudoReq {
  requestId: string
}

export interface SecretReq {
  envVar: string
  prompt: string
  requestId: string
}

export interface PromptOptimizationPreview {
  session_key: string
  original: string
  rewritten: string
  quality_before: number
  quality_after: number
  token_delta_pct: number
  model_profile: string
  template_name?: string | null
}

export interface PromptOptimizationReq {
  preview: PromptOptimizationPreview
  reason?: string
  status: 'bypass' | 'preview'
}

/** External password-manager unlock (1Password / Bitwarden) — masked master-password prompt. */
export interface VaultUnlockReq {
  backend: string
  displayName: string
  requestId: string
}

export interface PanelData {
  sections: PanelSection[]
  title: string
}

export interface PanelSection {
  items?: string[]
  rows?: [string, string][]
  text?: string
  title?: string
}

export interface SlashCatalog {
  canon: Record<string, string>
  categories: SlashCategory[]
  pairs: [string, string][]
  skillCount: number
  sub: Record<string, string[]>
}

export interface SlashCategory {
  name: string
  pairs: [string, string][]
}
