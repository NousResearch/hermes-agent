/**
 * Which emulator is hosting this TUI, when the host tells us.
 *
 * `rabbit dashboard` spawns the TUI behind a PTY and mirrors it into xterm.js
 * in the browser; the bridge sets RABBIT_PTY_HOST=dashboard on the child
 * (rabbit_cli/pty_bridge.py — keep the two constants in sync). Native
 * terminals never set it.
 */
export const PTY_HOST_ENV = 'RABBIT_PTY_HOST'
export const PTY_HOST_DASHBOARD = 'dashboard'

export const isDashboardHosted = (env: NodeJS.ProcessEnv = process.env): boolean =>
  env[PTY_HOST_ENV] === PTY_HOST_DASHBOARD
