import { homedir } from 'node:os'
import { join } from 'node:path'

/**
 * The Hermes home this process must write to, resolved the way the Python side
 * resolves it (`hermes_constants._get_platform_default_hermes_home()`): an
 * explicit `HERMES_HOME` wins, otherwise the PLATFORM default — `%LOCALAPPDATA%\
 * hermes` on native Windows, `~/.hermes` everywhere else.
 *
 * The launcher exports `HERMES_HOME` for a named profile, but a default-profile
 * launch leaves it unset, and `homedir()/.hermes` is not that default on
 * Windows. TUI state written to the wrong home (boot theme cache, crash
 * breadcrumbs, input history, heap dumps, user widgets) is then invisible to
 * everything that goes through `get_hermes_home()` — most visibly the panic log,
 * where the parent's `[tui-parent]` breadcrumbs and the gateway's own
 * `logs/tui_gateway_crash.log` entries split across two files that
 * `lib/parentLog.ts` exists to interleave by timestamp.
 *
 * Pure by construction: env, platform and the native home are parameters, so
 * callers and tests never inherit the host's answer by accident.
 */
export function resolveHermesHome(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
  nativeHome: string = homedir()
): string {
  const override = (env.HERMES_HOME ?? '').trim()

  if (override) {
    return override
  }

  if (platform === 'win32') {
    const localAppData = (env.LOCALAPPDATA ?? '').trim()

    return join(localAppData || join(nativeHome, 'AppData', 'Local'), 'hermes')
  }

  return join(nativeHome, '.hermes')
}
