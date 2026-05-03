export type ShellFamily = 'bash' | 'zsh' | 'fish' | 'posix-sh' | 'powershell' | 'tcsh' | 'unknown'

export type ShellSignalInput = {
  env: NodeJS.ProcessEnv
  executable?: string
  argv0?: string
  interactive?: boolean
}

export type ShellSignals = {
  family: ShellFamily
  executable?: string
  login: boolean
  interactive: boolean
  version?: string
  startupRiskHints: string[]
}

const shellRiskHints: Record<Exclude<ShellFamily, 'unknown'>, string> = {
  bash: '.bashrc can reset stty or TERM',
  zsh: '.zshenv always runs; check for output or stty mutations',
  fish: 'config.fish can run broadly; unguarded output breaks noninteractive commands',
  'posix-sh': 'minimal editor support; no reliable shell-layer bracketed paste',
  powershell: "PSReadLine doesn't apply when Hermes owns raw mode",
  tcsh: 'legacy line editing; startup files often contain old stty/tset calls'
}

const normalizeText = (value?: string): string | undefined => {
  const text = value?.trim()
  return text ? text : undefined
}

const normalizeShellName = (value?: string): string => {
  const text = normalizeText(value)
  if (!text) return ''

  const withoutLoginPrefix = text.startsWith('-') ? text.slice(1) : text
  const baseName = withoutLoginPrefix.split(/[\\/]/).filter(Boolean).pop()?.toLowerCase() ?? ''

  return baseName.endsWith('.exe') ? baseName.slice(0, -4) : baseName
}

const shellFamilyFromName = (name: string): Exclude<ShellFamily, 'unknown'> | undefined => {
  switch (name) {
    case 'bash':
    case 'rbash':
      return 'bash'
    case 'zsh':
    case 'rzsh':
      return 'zsh'
    case 'fish':
      return 'fish'
    case 'sh':
    case 'dash':
    case 'ash':
      return 'posix-sh'
    case 'pwsh':
    case 'powershell':
      return 'powershell'
    case 'tcsh':
    case 'csh':
      return 'tcsh'
    default:
      return undefined
  }
}

const shellVersionFromEnv = (env: NodeJS.ProcessEnv, family: Exclude<ShellFamily, 'unknown'>): string | undefined => {
  switch (family) {
    case 'bash':
      return normalizeText(env.BASH_VERSION)
    case 'zsh':
      return normalizeText(env.ZSH_VERSION)
    case 'fish':
      return normalizeText(env.FISH_VERSION)
    default:
      return undefined
  }
}

const inferInteractive = (env: NodeJS.ProcessEnv, login: boolean): boolean =>
  Boolean(login || normalizeText(env.PS1) || normalizeText(env.PROMPT))

export function detectShellSignals(input: ShellSignalInput): ShellSignals {
  const env = input.env
  const login = normalizeText(input.argv0)?.startsWith('-') ?? false
  const executableName = normalizeShellName(input.executable)

  const envFamily = normalizeText(env.BASH_VERSION)
    ? 'bash'
    : normalizeText(env.ZSH_VERSION)
      ? 'zsh'
      : normalizeText(env.FISH_VERSION)
        ? 'fish'
        : undefined

  const family = envFamily ?? shellFamilyFromName(executableName) ?? 'unknown'
  const version = family === 'unknown' ? undefined : shellVersionFromEnv(env, family)
  const interactive = input.interactive ?? inferInteractive(env, login)
  const executable = normalizeText(input.executable) ?? normalizeText(env.SHELL)

  return {
    family,
    executable,
    login,
    interactive,
    version,
    startupRiskHints: family === 'unknown' ? [] : [shellRiskHints[family]]
  }
}
