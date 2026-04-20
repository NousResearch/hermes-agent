import { coreCommands } from './commands/core.js'
import { opsCommands } from './commands/ops.js'
import { sessionCommands } from './commands/session.js'
import { setupCommands } from './commands/setup.js'
import type { SlashCommand } from './types.js'

export const SLASH_COMMANDS: SlashCommand[] = [...coreCommands, ...sessionCommands, ...opsCommands, ...setupCommands]

const byName = new Map<string, SlashCommand>(
  SLASH_COMMANDS.flatMap(cmd => [cmd.name, ...(cmd.aliases ?? [])].map(name => [name, cmd] as const))
)

export const findSlashCommand = (name: string) => byName.get(name.toLowerCase())
