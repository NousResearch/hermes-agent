import { detectShellSignals, type ShellSignals } from './shellSignals.js'

export type TerminalSignalInput = {
  env: NodeJS.ProcessEnv
  platform: NodeJS.Platform
  isStdinTty?: boolean
  isStdoutTty?: boolean
  shellExecutable?: string
  shellArgv0?: string
}

export type TerminalSignals = {
  platform: NodeJS.Platform
  isStdinTty: boolean
  isStdoutTty: boolean
  ssh: {
    hasSshConnection: boolean
    hasSshClient: boolean
    hasSshTty: boolean
    tty?: string
    connection?: string
  }
  multiplexer: {
    tmux: boolean
    screen: boolean
    zellij: boolean
    cy: boolean
  }
  env: {
    TERM?: string
    COLORTERM?: string
    TERM_PROGRAM?: string
    TERM_PROGRAM_VERSION?: string
    VTE_VERSION?: string
    WT_SESSION?: string
    KITTY_WINDOW_ID?: string
    WEZTERM_PANE?: string
    GHOSTTY_RESOURCES_DIR?: string
    KONSOLE_VERSION?: string
    ITERM_SESSION_ID?: string
    LC_TERMINAL?: string
    TERM_SESSION_ID?: string
  }
  shell: ShellSignals
}

const terminalEnvKeys = [
  'TERM',
  'COLORTERM',
  'TERM_PROGRAM',
  'TERM_PROGRAM_VERSION',
  'VTE_VERSION',
  'WT_SESSION',
  'KITTY_WINDOW_ID',
  'WEZTERM_PANE',
  'GHOSTTY_RESOURCES_DIR',
  'KONSOLE_VERSION',
  'ITERM_SESSION_ID',
  'LC_TERMINAL',
  'TERM_SESSION_ID'
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
  const sshConnection = pick(env, 'SSH_CONNECTION')
  const sshClient = pick(env, 'SSH_CLIENT')
  const sshTty = pick(env, 'SSH_TTY')

  const shell = detectShellSignals({
    env,
    executable: input.shellExecutable,
    argv0: input.shellArgv0,
    interactive: inferShellInteractive(input)
  })

  return {
    platform: input.platform,
    isStdinTty: input.isStdinTty === true,
    isStdoutTty: input.isStdoutTty === true,
    ssh: {
      hasSshConnection: sshConnection !== undefined,
      hasSshClient: sshClient !== undefined,
      hasSshTty: sshTty !== undefined,
      ...(sshTty !== undefined ? { tty: sshTty } : {}),
      ...(sshConnection !== undefined ? { connection: sshConnection } : {})
    },
    multiplexer: {
      tmux: pick(env, 'TMUX') !== undefined,
      screen: pick(env, 'STY') !== undefined,
      zellij: pick(env, 'ZELLIJ') !== undefined,
      cy: pick(env, 'CY') !== undefined
    },
    env: collectEnvSignals(env),
    shell
  }
}
