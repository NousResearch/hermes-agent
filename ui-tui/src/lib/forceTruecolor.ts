/**
 * Targeted 24-bit truecolor override before chalk / supports-color imports.
 *
 * macOS Terminal.app before Tahoe 26 does not support RGB SGR, so do not
 * infer truecolor from TERM_PROGRAM=Apple_Terminal. Users can still opt in
 * explicitly on terminals that support RGB but do not advertise COLORTERM.
 */

const TRUE_RE = /^(?:1|true|yes|on)$/i
const FALSE_RE = /^(?:0|false|no|off)$/i

export function shouldForceTruecolor(env: NodeJS.ProcessEnv = process.env): boolean {
  const override = (env.HERMES_TUI_TRUECOLOR ?? '').trim()

  if (FALSE_RE.test(override) || 'NO_COLOR' in env) {
    return false
  }

  return TRUE_RE.test(override)
}

if (shouldForceTruecolor()) {
  if (!process.env.COLORTERM) {
    process.env.COLORTERM = 'truecolor'
  }

  process.env.FORCE_COLOR = '3'
}

export {}
