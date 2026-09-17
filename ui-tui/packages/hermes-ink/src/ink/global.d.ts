/**
 * Ambient contracts for the hermes-ink package.
 *
 * The package's own source (previously excluded from typechecking by an
 * inherited `include` that resolved against the ui-tui root) typechecks
 * against the real runtime surface it uses, which spans three worlds:
 *
 *  1. Bun runtime globals (`Bun.semver`, `Bun.stringWidth`, `Bun.wrapAnsi`)
 *     used with pure-JS fallbacks when running under Node — declared as an
 *     `undefined`-able global so the `typeof Bun !== 'undefined'` guards in
 *     src stay truthful.
 *  2. The custom reconciler host elements (`<ink-box>`, `<ink-text>`, …)
 *     created by react-reconciler — declared as JSX intrinsics.
 *  3. Third-party modules that ship no types (bidi-js, react/compiler-runtime).
 */
import type { ReactNode } from 'react'

declare global {
  /**
   * Bun runtime globals, present only when executed by Bun. The source uses
   * `typeof Bun !== 'undefined'` guards with pure-JS fallbacks under Node, so
   * the honest type is `undefined`-able. Members limited to what src uses.
   */
  const Bun:
    | {
        readonly semver: {
          readonly order: (a: string, b: string) => -1 | 0 | 1
          readonly satisfies: (version: string, range: string) => boolean
        }
        readonly stringWidth: (
          input: string,
          options?: { readonly ambiguousIsNarrow?: boolean },
        ) => number
        readonly wrapAnsi: (
          input: string,
          columns: number,
          options?: { readonly hard?: boolean; readonly trim?: boolean; readonly wordWrap?: boolean },
        ) => string
        readonly indexOfFirstDifference?: (a: string, b: string) => number | undefined
      }
    | undefined
}

declare module 'react' {
  namespace JSX {
    /** Merged into React.JSX by declaration merging. */
    interface IntrinsicElements {
      /**
       * Host element created by the custom reconciler. Components spread
       * their (individually typed) props into it, so the element itself is
       * intentionally permissive — per-prop typing lives on the component
       * APIs that construct it.
       */
      'ink-box': any
      'ink-text': {
        readonly style?: unknown
        readonly textStyles?: unknown
        readonly children?: ReactNode
      }
      'ink-link': {
        readonly href: string
        readonly children?: ReactNode
      }
      'ink-raw-ansi': {
        readonly rawHeight?: number
        readonly rawText?: unknown
        readonly rawWidth?: number
      }
    }
  }
}

declare module 'react/compiler-runtime' {
  /**
   * Heterogeneous mutable memo cache — slots hold arbitrary values read back
   * after conditional writes, so the honest type is `any[]` (mismatched typing
   * here poisons every downstream compiler-generated binding with `unknown`).
   */
  export function c(size: number): any[]
}
