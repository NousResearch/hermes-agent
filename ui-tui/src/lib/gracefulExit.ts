interface SetupOptions {
  cleanups?: (() => Promise<void> | void)[]
  failsafeMs?: number
  onError?: (scope: 'uncaughtException' | 'unhandledRejection', err: unknown) => void
  onSignal?: (signal: NodeJS.Signals) => void
}

const SIGNAL_EXIT_CODE: Record<'SIGHUP' | 'SIGINT' | 'SIGTERM', number> = {
  SIGHUP: 129,
  SIGINT: 130,
  SIGTERM: 143
}

let wired = false

export function setupGracefulExit({ cleanups = [], failsafeMs = 4000, onError, onSignal }: SetupOptions = {}) {
  if (wired) {
    return
  }

  wired = true

  let shuttingDown = false

  const exit = (code: number, signal?: NodeJS.Signals) => {
    if (shuttingDown) {
      return
    }

    shuttingDown = true

    if (signal) {
      onSignal?.(signal)
    }

    setTimeout(() => process.exit(code), failsafeMs).unref?.()

    void Promise.allSettled(cleanups.map(fn => Promise.resolve().then(fn))).finally(() => process.exit(code))
  }

  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
    process.on(sig, () => exit(SIGNAL_EXIT_CODE[sig], sig))
  }

  process.on('uncaughtException', err => onError?.('uncaughtException', err))
  process.on('unhandledRejection', reason => onError?.('unhandledRejection', reason))
}

/**
 * Start a watchdog that detects when this process has been orphaned
 * (parent process exited, closing the stdin pipe or changing ppid).
 *
 * **Primary detection — stdin pipe closure (event-driven, zero overhead):**
 * When a terminal window is closed or the desktop app quits, the parent's
 * side of the stdin pipe closes.  We listen for `'end'` / `'close'` on
 * `process.stdin` — this fires immediately with no polling overhead.
 *
 * **Fallback — ppid polling:**
 * In exotic environments (containers, certain SSH setups) the stdin events
 * may not fire reliably.  As a safety net, we check `process.ppid` every
 * `fallbackIntervalMs` (default 30 s).  The interval is `.unref()`'d so
 * it does not keep the process alive on its own.
 *
 * @param onOrphaned  Called when orphaning is detected.  Receives a
 *                    human-readable reason string for logging.
 * @param fallbackIntervalMs  How often to poll ppid (default: 30 000).
 * @returns           A stop function that removes listeners and clears timers.
 */
export function startParentWatchdog(
  onOrphaned: (reason: string) => void,
  fallbackIntervalMs = 30_000
): () => void {
  // Not supported on Windows (ppid is unreliable, stdin events differ).
  if (process.platform === 'win32') {
    return () => {}
  }

  // Record the original parent PID at startup.  If it later changes
  // (the parent exited and we were reparented to init/launchd) or
  // becomes unresolvable, we know we are orphaned.
  const originalPpid = process.ppid

  // PID ≤ 1 means init / launchd — reparenting to it means the real
  // parent is gone.  On macOS orphaned processes go to PID 1 (launchd),
  // on Linux to PID 1 (init / systemd).
  if (originalPpid <= 1) {
    // Already orphaned at startup (unusual but possible if launched from
    // a short-lived wrapper).  Don't arm the watchdog — we'd fire immediately.
    return () => {}
  }

  let fired = false

  const fire = (reason: string) => {
    if (fired) {
      return
    }

    fired = true
    cleanup()
    onOrphaned(reason)
  }

  // --- Primary: stdin pipe closure detection ---
  // When the parent exits, its side of the stdin pipe closes.  Node's
  // Readable stream emits 'end' (no more data) and 'close' (resource
  // released).  Either event means the parent is gone.
  //
  // We listen on the raw file descriptor (fd 0) via `process.stdin`
  // because Ink's readline consumes the stream but does not prevent
  // the underlying 'end'/'close' from firing on the Readable itself.
  const onStdinEnd = () => fire('stdin pipe closed (parent exited)')
  const onStdinClose = () => fire('stdin stream closed')

  process.stdin.once('end', onStdinEnd)
  process.stdin.once('close', onStdinClose)

  // --- Fallback: ppid polling ---
  // Catches edge cases where stdin events don't fire (exotic container
  // setups, certain SSH configurations).  Runs at a longer interval
  // than the original 10 s because this is a safety net, not the
  // primary mechanism.
  const timer = setInterval(() => {
    try {
      const currentPpid = process.ppid

      // Reparented to init/launchd → parent exited.
      if (currentPpid <= 1) {
        fire(`parent exited (ppid changed from ${originalPpid} to ${currentPpid})`)
        return
      }

      // The ppid changed to a different non-init PID — unusual but
      // means the original parent is gone (e.g. inside a container
      // with a reaper).  Treat as orphaned.
      if (currentPpid !== originalPpid) {
        fire(`parent changed (ppid ${originalPpid} → ${currentPpid})`)
        return
      }

      // Original parent still alive — also verify with signal 0.
      // This catches the edge case where the PID was recycled by an
      // unrelated process.  Sending signal 0 to a process you don't
      // own may throw EPERM — that's fine (the process exists, we
      // just lack permission).  ESRCH means the process is gone.
      try {
        process.kill(originalPpid, 0)
      } catch (err: unknown) {
        if (isErrnoException(err) && err.code === 'ESRCH') {
          fire(`parent PID ${originalPpid} no longer exists (ESRCH)`)
        }
      }
    } catch {
      // process.ppid may be undefined in exotic environments — ignore.
    }
  }, fallbackIntervalMs)

  // Don't let the watchdog keep the process alive if everything else
  // has cleaned up.  The timer is the only thing holding the event loop
  // open in that case, and we want the process to exit naturally.
  timer.unref?.()

  function cleanup() {
    process.stdin.removeListener('end', onStdinEnd)
    process.stdin.removeListener('close', onStdinClose)
    clearInterval(timer)
  }

  return cleanup
}

function isErrnoException(err: unknown): err is NodeJS.ErrnoException {
  return err instanceof Error && 'code' in err
}
