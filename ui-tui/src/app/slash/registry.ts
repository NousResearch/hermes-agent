import { coreCommands } from './commands/core.js'
import { debugCommands } from './commands/debug.js'
import { opsCommands } from './commands/ops.js'
import { sessionCommands } from './commands/session.js'
import { setupCommands } from './commands/setup.js'
import { subscriptionCommands } from './commands/subscription.js'
import { topupCommands } from './commands/topup.js'
import { wakeCommands } from './commands/wake.js'
import type { SlashCommand } from './types.js'

const employeeExcludedCommands = new Set([
  'skills', 'reload-skills', 'learn', 'bundles', 'curator', 'kanban', 'cron', 'blueprint', 'suggestions', 'personality'
])

export const SLASH_COMMANDS: SlashCommand[] = [
  ...coreCommands,
  ...topupCommands,
  ...sessionCommands,
  ...subscriptionCommands,
  ...opsCommands,
  ...wakeCommands,
  ...setupCommands,
  ...debugCommands
].filter(command => !employeeExcludedCommands.has(command.name))

const byName = new Map<string, SlashCommand>(
  SLASH_COMMANDS.flatMap(cmd => [cmd.name, ...(cmd.aliases ?? [])].map(name => [name, cmd] as const))
)

export const findSlashCommand = (name: string) => byName.get(name.toLowerCase())
