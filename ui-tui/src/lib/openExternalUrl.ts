import { spawn, type SpawnOptions } from 'node:child_process'
import { platform } from 'node:os'

/**
 * Opens an external URL in the user's default browser/handler.
 *
 * Wired into the Ink instance via `onHyperlinkClick` in entry.tsx, so any
 * mouse click on a `<Link>` cell (or a row containing a plain-text URL the
 * renderer detected) goes here. Mouse tracking inside the TUI prevents
 * Terminal.app's native Cmd+click from firing — the click is captured
 * before the terminal application sees it — so we have to handle the open
 * ourselves.
 *
 * Safety:
 * - http(s) only. Anything else (`file:`, `data:`, `javascript:`, etc.) is
 *   rejected — a hostile model could otherwise emit `<Link url="file:///">`
 *   and trick a click into running an arbitrary local handler.
 * - Hostname is parsed via `URL`; only well-formed URLs are forwarded.
 * - Spawned via `child_process.spawn` with arg array (no shell), so a URL
 *   containing shell metacharacters (`;`, `&`, backticks) cannot be
 *   interpreted as a command.
 *
 * Returns `true` if the spawn was attempted, `false` if the URL was rejected.
 */
export function openExternalUrl(rawUrl: string, dependencies: OpenDependencies = {}): boolean {
  const url = parseSafeUrl(rawUrl)

  if (!url) {
    return false
  }

  const spawnFn = dependencies.spawn ?? spawn
  const platformId = dependencies.platform?.() ?? platform()

  const command = openCommand(platformId)

  if (!command) {
    return false
  }

  try {
    const child = spawnFn(command.command, [...command.args, url.toString()], {
      // Detach so closing the TUI later doesn't kill the browser process,
      // and ignore stdio so we don't leak FDs into our raw-mode terminal.
      // Without `ignore` here, Chrome's stderr can land in the alt screen.
      detached: true,
      stdio: 'ignore'
    } satisfies SpawnOptions)
    child.unref()

    return true
  } catch {
    // spawn can throw synchronously on unusable PATHs (e.g. WSL without an
    // explorer.exe shim). Treat it as a no-op rather than crashing the TUI.
    return false
  }
}

export type OpenDependencies = {
  spawn?: typeof spawn
  platform?: () => string
}

/**
 * Validate and normalize a URL for opening externally.
 * Exported for testing.
 */
export function parseSafeUrl(value: string): null | URL {
  if (!value || typeof value !== 'string') {
    return null
  }

  let parsed: URL

  try {
    parsed = new URL(value)
  } catch {
    return null
  }

  // http(s) only — opening file://, data:, javascript:, vbscript:, etc.
  // would let a malicious model run a local handler with attacker-controlled
  // input on a single click.
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    return null
  }

  // Reject empty or all-whitespace hostnames defensively. URL parsing
  // accepts URLs like 'http:///foo' on some Node versions; we don't want
  // to forward those to `open`.
  if (!parsed.hostname.trim()) {
    return null
  }

  return parsed
}

type OpenCommand = { command: string; args: readonly string[] }

/**
 * Per-platform open command. We deliberately avoid `cmd.exe /c start` on
 * Windows even though it's the canonical example, because `start` is a cmd
 * builtin: the URL string is reparsed by cmd's command-line tokenizer and
 * characters like `&`, `|`, `^`, `<`, `>` either break the command or get
 * interpreted as additional commands. That undermines the protocol
 * allowlist's safety story and also breaks plain http(s) URLs with `&` in
 * query strings. `explorer.exe <url>` is the safe, non-shell alternative —
 * it invokes the registered protocol handler for http(s) without going
 * through cmd. Linux/BSD use `xdg-open` directly with no shell wrapping.
 */
export function openCommand(platformId: string): OpenCommand | null {
  if (platformId === 'darwin') {
    return { command: 'open', args: [] }
  }

  if (platformId === 'win32') {
    return { command: 'explorer.exe', args: [] }
  }

  return { command: 'xdg-open', args: [] }
}
