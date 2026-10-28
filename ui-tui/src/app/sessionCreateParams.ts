import { STARTUP_TEMPORARY, STARTUP_WORKSPACE_CWD } from '../config/env.js'

/** `session.create` params for a fresh TUI session. `hermes --temporary` makes every session this
 *  process creates a temporary chat (never listed or saved, no memory; deleted when it closes). */
export function sessionCreateParams(
  cols: number,
  startup: { temporary: boolean; workspaceCwd: string } = {
    temporary: STARTUP_TEMPORARY,
    workspaceCwd: STARTUP_WORKSPACE_CWD
  }
): Record<string, unknown> {
  return {
    cols,
    ...(startup.workspaceCwd ? { cwd: startup.workspaceCwd } : {}),
    ...(startup.temporary ? { temporary: true } : {})
  }
}

/** The canonical gateway's `session.create` option for a `hermes --temporary` launch (empty otherwise). */
export function temporaryCreateOption(temporary: boolean = STARTUP_TEMPORARY): Record<string, unknown> {
  return temporary ? { temporary: true } : {}
}
