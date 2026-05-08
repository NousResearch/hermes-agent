import { detectShellSignals, type ShellSignals } from './shellSignals.js'

export type TerminalSignalInput = {
  env: NodeJS.ProcessEnv
  isStdinTty?: boolean
  isStdoutTty?: boolean
  platform: NodeJS.Platform
  shellArgv0?: string
  shellExecutable?: string
}

export type TerminalSignals = {
  env: {
    COLORTERM?: string
    GHOSTTY_RESOURCES_DIR?: string
    ITERM_SESSION_ID?: string
    KITTY_WINDOW_ID?: string
    KONSOLE_VERSION?: string
    LC_TERMINAL?: string
    TERM?: string
    TERM_PROGRAM?: string
    TERM_PROGRAM_VERSION?: string
    TERM_SESSION_ID?: string
    VTE_VERSION?: string
    WEZTERM_PANE?: string
    WT_SESSION?: string
  }
  isStdinTty: boolean
  isStdoutTty: boolean
  multiplexer: {
    cy: boolean
    screen: boolean
    tmux: boolean
    zellij: boolean
  }
  platform: NodeJS.Platform
  shell: ShellSignals
  ssh: {
    client?: string
    connection?: string
    hasSshClient: boolean
    hasSshConnection: boolean
    hasSshTty: boolean
    tty?: string
  }
}

const terminalEnvKeys = [
  'COLORTERM',
  'GHOSTTY_RESOURCES_DIR',
  'ITERM_SESSION_ID',
  'KITTY_WINDOW_ID',
  'KONSOLE_VERSION',
  'LC_TERMINAL',
  'TERM',
  'TERM_PROGRAM',
  'TERM_PROGRAM_VERSION',
  'TERM_SESSION_ID',
  'VTE_VERSION',
  'WEZTERM_PANE',
  'WT_SESSION'
] as const

type TerminalEnvKey = (typeof terminalEnvKeys)[number]

const pick = (env: NodeJS.ProcessEnv, key: string): string | undefined => {
  const value = env[key]

  return typeof value === 'string' && value.length > 0 ? value : undefined
}

const collectEnvSignals = (env: NodeJS.ProcessEnv): TerminalSignals['env'] => {
  const signals: Partial<Record<TerminalEnvKey, string>> = {}

  for (const key of terminalEnvKeys) {
    const value = pick(env, key)

    if (value !== undefined) {
      signals[key] = value
    }
  }

  return signals as TerminalSignals['env']
}

const inferShellInteractive = (input: TerminalSignalInput): boolean | undefined => {
  if (input.isStdinTty === undefined && input.isStdoutTty === undefined) {
    return undefined
  }

  return input.isStdinTty === true || input.isStdoutTty === true
}

export function collectTerminalSignals(input: TerminalSignalInput): TerminalSignals {
  const env = input.env
  const sshClient = pick(env, 'SSH_CLIENT')
  const sshConnection = pick(env, 'SSH_CONNECTION')
  const sshTty = pick(env, 'SSH_TTY')

  const shell = detectShellSignals({
    argv0: input.shellArgv0,
    env,
    executable: input.shellExecutable,
    interactive: inferShellInteractive(input)
  })

  return {
    env: collectEnvSignals(env),
    isStdinTty: input.isStdinTty === true,
    isStdoutTty: input.isStdoutTty === true,
    multiplexer: {
      cy: pick(env, 'CY') !== undefined,
      screen: pick(env, 'STY') !== undefined,
      tmux: pick(env, 'TMUX') !== undefined,
      zellij: pick(env, 'ZELLIJ') !== undefined
    },
    platform: input.platform,
    shell,
    ssh: {
      hasSshClient: sshClient !== undefined,
      hasSshConnection: sshConnection !== undefined,
      hasSshTty: sshTty !== undefined,
      ...(sshClient !== undefined ? { client: sshClient } : {}),
      ...(sshConnection !== undefined ? { connection: sshConnection } : {}),
      ...(sshTty !== undefined ? { tty: sshTty } : {})
    }
  }
}
