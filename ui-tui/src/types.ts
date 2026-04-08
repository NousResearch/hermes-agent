export interface ActiveTool {
  id: string
  name: string
  context?: string
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
  kind?: 'intro' | 'tool-active'
  info?: SessionInfo
  toolId?: string
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

/** From `commands.catalog` — mirrors hermes_cli.commands COMMANDS + SUBCOMMANDS + skills. */
export interface SlashCatalog {
  canon: Record<string, string>
  pairs: [string, string][]
  sub: Record<string, string[]>
}
