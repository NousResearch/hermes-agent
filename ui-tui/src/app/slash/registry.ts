import { coreCommands } from './commands/core.js'
import { opsCommands } from './commands/ops.js'
import { sessionCommands } from './commands/session.js'
import type { SlashCommand } from './types.js'

export const SLASH_COMMANDS: SlashCommand[] = [...coreCommands, ...sessionCommands, ...opsCommands]

const byName = new Map<string, SlashCommand>()

for (const cmd of SLASH_COMMANDS) {
  byName.set(cmd.name, cmd)

  for (const alias of cmd.aliases ?? []) {
    byName.set(alias, cmd)
  }
}

export const findSlashCommand = (name: string): SlashCommand | undefined => byName.get(name.toLowerCase())
