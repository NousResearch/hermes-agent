export interface ActiveTool {
  id: string
  name: string
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
}

export type Role = 'assistant' | 'system' | 'tool' | 'user'

export interface SessionInfo {
  model: string
  skills: Record<string, string[]>
  tools: Record<string, string[]>
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
