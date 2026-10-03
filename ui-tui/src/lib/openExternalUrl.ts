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
 * Returns `true` if the spawn was attempted, `false` if the open could
 * not proceed — covers (a) URL rejected by `parseSafeUrl` (non-http(s),
 * malformed, etc.), (b) no known opener for the current platform
 * (`openCommand` returned null), or (c) `spawn()` threw synchronously
 * before the child was created. Async failures after spawn (`'error'`
 * event because the binary couldn't exec) still return `true` because
 * the spawn was attempted — the no-op error listener absorbs the event
 * so the TUI doesn't crash, and the user just doesn't see their browser
 * pop.
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

    // Async failure path: spawn returns a ChildProcess synchronously even
    // when the binary is missing (ENOENT on `xdg-open` / `explorer.exe`),
    // unreachable (EACCES), or otherwise unusable — the failure surfaces
    // later as an 'error' event. Without a handler, an unhandled 'error'
    // on an EventEmitter crashes Node, which would tear down the whole
    // TUI. Attach a no-op listener BEFORE unref() so the event has a
    // consumer; we already returned `true` synchronously, so the user
    // just won't see their browser open — same as if the URL had been
    // rejected upstream.
    child.once('error', () => {
      // Intentional no-op. The TUI keeps running; user gets no browser
      // pop, which is the failure mode we promised in the doc comment.
    })

    child.unref()

    return true
  } catch {
    // spawn can also throw synchronously on argv-validation failures
    // (e.g. NUL in the path). Treat it as a no-op rather than crashing.
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
 *
 * Returns null for platforms where we don't know a safe opener (e.g. `aix`,
 * `sunos`, `cygwin`). The caller's `if (!command) return false` path then
 * surfaces "no opener" instead of optimistically trying `xdg-open` on a
 * platform that probably doesn't have it.
 */
export function openCommand(platformId: string): OpenCommand | null {
  if (platformId === 'darwin') {
    return { command: 'open', args: [] }
  }

  if (platformId === 'win32') {
    return { command: 'explorer.exe', args: [] }
  }

  // Linux + the BSD family ship xdg-open via xdg-utils. Everything else
  // (aix, sunos, cygwin, haiku, etc.) returns null so openExternalUrl's
  // command-not-found fallback fires honestly.
  const XDG_OPEN_PLATFORMS = new Set(['linux', 'freebsd', 'openbsd', 'netbsd', 'dragonfly'])

  if (XDG_OPEN_PLATFORMS.has(platformId)) {
    return { command: 'xdg-open', args: [] }
  }

  return null
}
