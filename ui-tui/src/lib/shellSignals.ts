export type ShellFamily = 'bash' | 'zsh' | 'fish' | 'posix-sh' | 'powershell' | 'tcsh' | 'unknown'

export type ShellSignalInput = {
  argv0?: string
  env: NodeJS.ProcessEnv
  executable?: string
  interactive?: boolean
}

export type ShellSignals = {
  executable?: string
  family: ShellFamily
  interactive: boolean
  login: boolean
  version?: string
}

const normalizeText = (value?: string): string | undefined => {
  const text = value?.trim()

  return text ? text : undefined
}

const normalizeShellName = (value?: string): string => {
  const text = normalizeText(value)

  if (!text) {return ''}

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

    case 'ash':

    case 'dash':

    case 'sh':
      return 'posix-sh'

    case 'powershell':

    case 'pwsh':
      return 'powershell'

    case 'csh':

    case 'tcsh':
      return 'tcsh'

    default:
      return undefined
  }
}

const shellVersionFromEnv = (env: NodeJS.ProcessEnv, family: Exclude<ShellFamily, 'unknown'>): string | undefined => {
  switch (family) {
    case 'bash':
      return normalizeText(env.BASH_VERSION)

    case 'fish':
      return normalizeText(env.FISH_VERSION)

    case 'zsh':
      return normalizeText(env.ZSH_VERSION)

    default:
      return undefined
  }
}

const inferInteractive = (env: NodeJS.ProcessEnv, login: boolean): boolean =>
  Boolean(login || normalizeText(env.PS1) || normalizeText(env.PROMPT))

export function detectShellSignals(input: ShellSignalInput): ShellSignals {
  const env = input.env ?? process.env
  const login = normalizeText(input.argv0)?.startsWith('-') ?? false
  const executable = normalizeText(input.executable) ?? normalizeText(env.SHELL)
  const executableName = normalizeShellName(executable)

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

  return {
    executable,
    family,
    interactive,
    login,
    version
  }
}
