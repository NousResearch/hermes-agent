import { coreCommands } from './commands/core.js';
import { opsCommands } from './commands/ops.js';
import { sessionCommands } from './commands/session.js';
import { setupCommands } from './commands/setup.js';
export const SLASH_COMMANDS = [...coreCommands, ...sessionCommands, ...opsCommands, ...setupCommands];
const byName = new Map(SLASH_COMMANDS.flatMap(cmd => [cmd.name, ...(cmd.aliases ?? [])].map(name => [name, cmd])));
export const findSlashCommand = (name) => byName.get(name.toLowerCase());
