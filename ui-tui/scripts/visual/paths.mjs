// Shared output location for the visual harness. Hardcoded '/tmp/...' paths
// resolve to a drive-root like C:\tmp on native Windows (and fail when the
// directory doesn't exist) — os.tmpdir() is the platform-neutral answer.
// Both render.tsx and shot.mjs derive the same directory from here;
// HERMES_TUI_VISUAL_DIR overrides it for CI or side-by-side runs.
import { tmpdir } from 'os'
import { join } from 'path'

export function visualOutDir() {
  return process.env.HERMES_TUI_VISUAL_DIR || join(tmpdir(), 'hermes-tui-visual')
}
