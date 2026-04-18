export interface ActiveTool {
  context?: string
  id: string
  name: string
  startedAt?: number
}

export interface ActivityItem {
  id: number
  text: string
  tone: 'error' | 'info' | 'warn'
}

export interface SubagentProgress {
  durationSeconds?: number
  goal: string
  id: string
  index: number
  notes: string[]
  status: 'completed' | 'failed' | 'interrupted' | 'running'
  summary?: string
  taskCount: number
  thinking: string[]
  tools: string[]
}

export interface ApprovalReq {
  command: string
  description: string
}

export interface ClarifyReq {
  choices: string[] | null
  question: string
  requestId: string
}

export interface Msg {
  info?: SessionInfo
  kind?: 'intro' | 'panel' | 'slash' | 'trail'
  panelData?: PanelData
  role: Role
  text: string
  thinking?: string
  thinkingTokens?: number
  toolTokens?: number
  tools?: string[]
}

export type Role = 'assistant' | 'system' | 'tool' | 'user'
export type DetailsMode = 'hidden' | 'collapsed' | 'expanded'
export type ThinkingMode = 'collapsed' | 'truncated' | 'full'

export interface SessionInfo {
  cwd?: string
  model: string
  release_date?: string
  skills: Record<string, string[]>
  tools: Record<string, string[]>
  update_behind?: number | null
  update_command?: string
  usage?: Usage
  version?: string
}

export interface Usage {
  calls: number
  context_max?: number
  context_percent?: number
  context_used?: number
  cost_usd?: number
  input: number
  output: number
  total: number
}

export interface SudoReq {
  requestId: string
}

export interface SecretReq {
  envVar: string
  prompt: string
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
