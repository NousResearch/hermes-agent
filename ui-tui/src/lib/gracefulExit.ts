interface SetupOptions {
  cleanups?: (() => Promise<void> | void)[]
  failsafeMs?: number
  ignoredSignals?: GracefulSignal[]
  onError?: (scope: 'uncaughtException' | 'unhandledRejection', err: unknown) => void
  onSignal?: (signal: NodeJS.Signals) => void
}

export type GracefulSignal = 'SIGHUP' | 'SIGINT' | 'SIGTERM'

const SIGNALS: readonly GracefulSignal[] = ['SIGINT', 'SIGTERM', 'SIGHUP']

const SIGNAL_EXIT_CODE: Record<GracefulSignal, number> = {
  SIGHUP: 129,
  SIGINT: 130,
  SIGTERM: 143
}

let wired = false

export const shouldExitForSignal = (signal: GracefulSignal, ignoredSignals: readonly GracefulSignal[] = []) =>
  !ignoredSignals.includes(signal)

export function setupGracefulExit({
  cleanups = [],
  failsafeMs = 4000,
  ignoredSignals = [],
  onError,
  onSignal
}: SetupOptions = {}) {
  if (wired) {
    return
  }

  wired = true

  let shuttingDown = false

  const exit = (code: number, beforeCleanup?: () => void) => {
    if (shuttingDown) {
      return
    }

    shuttingDown = true

    // Arm the hard-exit backstop before diagnostics or other user-provided
    // hooks. A hook is allowed to fail, but it must never strand the process
    // with an uncaught fatal error and a still-running gateway child.
    // Keep this timer referenced. A pending Promise does not keep Node alive,
    // so unref'ing the only backstop can make a fatal process exit naturally
    // with status 0 while cleanup is still hung.
    setTimeout(() => process.exit(code), failsafeMs)

    try {
      beforeCleanup?.()
    } catch {
      // Best-effort diagnostics only. Cleanup and the requested exit code are
      // the lifecycle contract, even when a hook throws.
    }

    void Promise.allSettled(cleanups.map(fn => Promise.resolve().then(fn))).finally(() => process.exit(code))
  }

  for (const sig of SIGNALS) {
    process.on(sig, () => {
      if (!shouldExitForSignal(sig, ignoredSignals)) {
        return
      }

      exit(SIGNAL_EXIT_CODE[sig], () => onSignal?.(sig))
    })
  }

  process.on('uncaughtException', err => exit(1, () => onError?.('uncaughtException', err)))
  process.on('unhandledRejection', reason => exit(1, () => onError?.('unhandledRejection', reason)))
}
