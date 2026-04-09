export interface ActiveTool {
  id: string
  name: string
  context?: string
}

export interface ActivityItem {
  id: number
  text: string
  tone: 'error' | 'info' | 'warn'
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
  role: Role
  text: string
  kind?: 'intro' | 'slash'
  info?: SessionInfo
  thinking?: string
  tools?: string[]
}

export type Role = 'assistant' | 'system' | 'tool' | 'user'

export interface SessionInfo {
  cwd?: string
  model: string
  release_date?: string
  skills: Record<string, string[]>
  tools: Record<string, string[]>
  update_behind?: number | null
  update_command?: string
  version?: string
}

export interface Usage {
  calls: number
  context_max?: number
  context_percent?: number
  context_used?: number
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

export type PasteKind = 'code' | 'log' | 'text'
export type PasteMode = 'attach' | 'excerpt' | 'inline'

export interface PendingPaste {
  charCount: number
  createdAt: number
  id: number
  kind: PasteKind
  lineCount: number
  mode: PasteMode
  text: string
}

export interface SlashCatalog {
  canon: Record<string, string>
  pairs: [string, string][]
  sub: Record<string, string[]>
}
