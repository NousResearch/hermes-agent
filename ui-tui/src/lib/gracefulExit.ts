type Cleanup = () => Promise<void> | void

interface SetupOptions {
  cleanups?: Cleanup[]
  failsafeMs?: number
  onError?: (scope: 'uncaughtException' | 'unhandledRejection', err: unknown) => void
  onSignal?: (signal: NodeJS.Signals) => void
}

const DEFAULT_FAILSAFE_MS = 4000

let wired = false

export function setupGracefulExit({
  cleanups = [],
  failsafeMs = DEFAULT_FAILSAFE_MS,
  onError,
  onSignal
}: SetupOptions = {}) {
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

    const failsafe = setTimeout(() => process.exit(code), failsafeMs)

    failsafe.unref?.()

    void Promise.allSettled(cleanups.map(fn => Promise.resolve().then(fn)))
      .catch(() => {})
      .finally(() => process.exit(code))
  }

  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
    process.on(sig, () => exit(sig === 'SIGINT' ? 130 : sig === 'SIGTERM' ? 143 : 129, sig))
  }

  process.on('uncaughtException', err => {
    onError?.('uncaughtException', err)
  })

  process.on('unhandledRejection', reason => {
    onError?.('unhandledRejection', reason)
  })
}

export function forceExit(code = 0) {
  process.exit(code)
}
