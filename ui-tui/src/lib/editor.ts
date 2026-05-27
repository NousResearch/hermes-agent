import { accessSync, constants } from 'node:fs'
import { delimiter, join } from 'node:path'

const VSCODE_COMMANDS = new Set(['code', 'code-insiders', 'codium', 'vscodium'])
const VSCODE_FALLBACKS = ['code', 'code-insiders', 'codium', 'vscodium']

/** Editor fallback chain when neither $VISUAL nor $EDITOR is set. */
const FALLBACKS = ['editor', 'nano', 'pico', 'vi', 'emacs']

const isExecutable = (path: string): boolean => {
  try {
    accessSync(path, constants.X_OK)

    return true
  } catch {
    return false
  }
}

const commandName = (cmd: string): string => cmd.split(/[\\/]/).pop()?.replace(/\.(cmd|exe)$/i, '') ?? cmd

const findExecutable = (names: string[], env: NodeJS.ProcessEnv): null | string => {
  const dirs = (env.PATH ?? '').split(delimiter).filter(Boolean)

  return names.flatMap(name => dirs.map(d => join(d, name))).find(isExecutable) ?? null
}

const withBlockingArgs = (argv: string[]): string[] => {
  const [cmd, ...args] = argv

  if (!cmd || !VSCODE_COMMANDS.has(commandName(cmd))) {
    return argv
  }

  return args.includes('--wait') ? argv : [cmd, ...args, '--wait']
}

/**
 * Resolve the editor invocation argv (without the file argument).
 *
 *   1. $VISUAL / $EDITOR, shell-tokenized so `EDITOR="code --wait"` works
 *   2. on POSIX: VS Code (`code --wait`) when it is resolvable on $PATH
 *   3. on POSIX: first FALLBACKS entry resolvable on $PATH
 *   4. on Windows: `notepad.exe`
 *   5. literal `['vi']` as the last-resort POSIX floor
 */
export const resolveEditor = (
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): string[] => {
  const explicit = env.VISUAL ?? env.EDITOR

  if (explicit?.trim()) {
    return withBlockingArgs(explicit.trim().split(/\s+/))
  }

  if (platform === 'win32') {
    return ['notepad.exe']
  }

  const vscode = findExecutable(VSCODE_FALLBACKS, env)

  if (vscode) {
    return withBlockingArgs([vscode])
  }

  const found = findExecutable(FALLBACKS, env)

  return [found ?? 'vi']
}
