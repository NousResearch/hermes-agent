/** Process-level fallback for stdin EOF (terminal-driver Ctrl+D outside raw
 * mode, parent process closing the pipe). The Ink layer detects
 * `readableEnded` inside its own 'readable' handler, but that handler can be
 * detached mid-teardown or wedged after a raw-mode drop; the stream's 'end'
 * event still fires exactly once after the last byte is consumed, so it is
 * the reliable backstop (#24377). */

export interface StdinEofExitHooks {
  exit: (code: number) => void
  killGateway: () => void
  recordLifecycle: (line: string) => void
  resetModes: () => void
  stopMonitor: () => void
}

export function installStdinEofExit(stdin: NodeJS.ReadStream, hooks: StdinEofExitHooks): void {
  stdin.on('end', () => {
    // Same ordering as the memory-critical exit path in entry.tsx: leave a
    // breadcrumb first so a post-mortem can tell EOF apart from a signal
    // kill, restore the terminal before the gateway teardown can block, and
    // only then exit.
    hooks.recordLifecycle('stdin EOF → clean exit')
    hooks.stopMonitor()
    hooks.resetModes()
    hooks.killGateway()
    hooks.exit(0)
  })
}
