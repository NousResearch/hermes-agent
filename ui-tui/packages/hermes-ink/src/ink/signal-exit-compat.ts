import * as signalExit from 'signal-exit'

export type SignalExitHandler = (code: number | null | undefined, signal: NodeJS.Signals | null) => true | void
export type OnProcessExit = (handler: SignalExitHandler, options?: { alwaysLast?: boolean }) => () => void

type SignalExitCompatModule = {
  default?: unknown
  onExit?: unknown
}

export function resolveOnProcessExit(signalExitModule: SignalExitCompatModule): OnProcessExit {
  if (typeof signalExitModule.onExit === 'function') {
    return signalExitModule.onExit as OnProcessExit
  }

  if (typeof signalExitModule.default === 'function') {
    return signalExitModule.default as OnProcessExit
  }

  throw new TypeError('Unsupported signal-exit export shape')
}

export const onProcessExit = resolveOnProcessExit(signalExit)
