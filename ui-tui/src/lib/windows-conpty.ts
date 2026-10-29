/**
 * Detect native Windows ConPTY hosts where DEC mouse tracking never
 * arrives and AlternateScreen therefore swallows wheel + native selection.
 *
 * Git Bash/MSYS mintty outside Windows Terminal has a real PTY and keeps
 * the existing AlternateScreen default. A WT tab (including git-bash inside
 * WT) sets WT_SESSION and still goes through ConPTY.
 */
export const isWindowsConptyTuiMode = (
  env: NodeJS.ProcessEnv = process.env,
  platform: string = process.platform
): boolean => {
  if (platform !== 'win32') {
    return false
  }

  const msystem = String(env.MSYSTEM ?? '').trim()
  const wtSession = String(env.WT_SESSION ?? '').trim()

  if (msystem && !wtSession) {
    return false
  }

  return true
}
