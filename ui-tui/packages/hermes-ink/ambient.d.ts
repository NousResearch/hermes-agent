/// <reference types="react" />

declare module 'react/compiler-runtime' {
  export function c(size: number): any[]
}

declare module 'bidi-js' {
  const bidiFactory: () => Record<string, any>
  export default bidiFactory
}

declare module 'stack-utils' {
  class StackUtils {
    static nodeInternals(): RegExp[]
    constructor(opts?: { cwd?: string; internals?: RegExp[] })
    clean(stack: string | undefined): string | undefined
    parseLine(line: string): { file?: string; line?: number; column?: number; function?: string } | undefined
  }
  export default StackUtils
}

declare module 'react-reconciler' {
  export type FiberRoot = unknown
  const createReconciler: any
  export default createReconciler
}

declare module 'react-reconciler/constants.js' {
  export const ConcurrentRoot: number
  export const LegacyRoot: number
  export const DiscreteEventPriority: symbol | number
  export const ContinuousEventPriority: symbol | number
  export const DefaultEventPriority: symbol | number
  export const NoEventPriority: symbol | number
}

declare module 'lodash-es/noop.js' {
  const noop: (...args: unknown[]) => void
  export default noop
}

declare module 'lodash-es/throttle.js' {
  function throttle<T extends (...args: unknown[]) => unknown>(
    fn: T,
    wait?: number,
    opts?: { leading?: boolean; trailing?: boolean }
  ): T & { cancel(): void; flush(): void }
  export default throttle
}

declare module 'semver' {
  export function coerce(version: string | number | null | undefined): { version: string } | null
  export function gt(a: string, b: string, opts?: { loose?: boolean }): boolean
  export function gte(a: string, b: string, opts?: { loose?: boolean }): boolean
  export function lt(a: string, b: string, opts?: { loose?: boolean }): boolean
  export function lte(a: string, b: string, opts?: { loose?: boolean }): boolean
  export function satisfies(version: string, range: string, opts?: { loose?: boolean }): boolean
  export function compare(a: string, b: string, opts?: { loose?: boolean }): number
}

interface BunSemver {
  order(a: string, b: string): -1 | 0 | 1
  satisfies(version: string, range: string): boolean
}

interface BunRuntime {
  stringWidth(s: string, opts?: { ambiguousIsNarrow?: boolean }): number
  semver: BunSemver
  wrapAnsi?(input: string, columns: number, options?: { hard?: boolean; wordWrap?: boolean; trim?: boolean }): string
}

declare var Bun: BunRuntime | undefined

declare namespace React {
  namespace JSX {
    interface IntrinsicElements {
      'ink-box': Record<string, unknown>
      'ink-text': Record<string, unknown>
      'ink-link': Record<string, unknown>
      'ink-raw-ansi': Record<string, unknown>
    }
  }
}
