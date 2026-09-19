/**
 * Typed wrappers around Node's process signal-handler API.
 *
 * Why this exists: overload resolution for `process.on('SIGCONT', ...)`
 * depends on which ambient augmentation of `NodeJS.ProcessSignals` wins.
 * The package pins `types: ["node"]` in tsconfig, but an ambient `@types/bun`
 * augmentation has broken this exact overload before (the `process.on`
 * overload set collapsed and every SIGCONT registration became a type
 * error). Routing every handler registration through this module means a
 * future ambient-type conflict is fixed in one file instead of at each call
 * site — and the signal surface of the runtime stays grep-able.
 *
 * Only signals the ink runtime actually observes belong here.
 */

/** Signals the ink runtime registers handlers for. */
export type InkSignal = 'SIGCONT'

/**
 * Zero-argument listener shape the runtime uses. Node delivers the signal
 * name as the first argument, but every ink handler ignores it, so the
 * wrapper intentionally narrows the contract to what callers may rely on.
 */
export type SignalListener = () => void

/**
 * The registration target. Defaults to the real `process`; tests inject a
 * recorder instead of registering real OS-level handlers (SIGSTOP/SIGCONT
 * round-trips are not portable to Windows hosts).
 */
export interface SignalTarget {
  on(signal: InkSignal, listener: SignalListener): unknown
  off(signal: InkSignal, listener: SignalListener): unknown
}

const defaultTarget: SignalTarget = process

/** Register a signal handler. Delegates to `process.on`. */
export function onSignal(signal: InkSignal, listener: SignalListener, target: SignalTarget = defaultTarget): void {
  target.on(signal, listener)
}

/** Unregister a previously registered handler. Delegates to `process.off`. */
export function offSignal(signal: InkSignal, listener: SignalListener, target: SignalTarget = defaultTarget): void {
  target.off(signal, listener)
}
