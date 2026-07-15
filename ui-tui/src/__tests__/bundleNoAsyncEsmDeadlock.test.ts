/**
 * Bundle-shape regression for issue #31227.
 *
 * The dashboard TUI ships as a single esbuild-bundled `dist/entry.js`.
 * When the bundle contains an `async`-init `__esm` wrapper that participates
 * in a circular module graph, esbuild's lightweight init helper deadlocks
 * the top-level `await Promise.all([...])` in src/entry.tsx — the user
 * sees only 141 bytes of ANSI reset sequences and a blank screen forever.
 *
 * Root cause: re-exporting `ink-text-input` from `@hermes/ink`'s
 * entry-exports drags the upstream `ink` package into the bundle. That
 * `ink` graph and our in-tree `@hermes/ink` graph reference each other
 * via React/`ink-text-input`, producing the circular async cycle that
 * `__esm` cannot resolve.
 *
 * These tests guard the two structural properties that, together,
 * keep the bundle deadlock-free:
 *
 *  1. No `async` `__esm` modules in the bundle. As long as every init
 *     runs synchronously, `__esm`'s closure-capture quirk is irrelevant.
 *  2. No `ink-text-input` / `node_modules/ink/build` modules in the
 *     bundle. Their absence is what makes #1 hold; if a future commit
 *     re-introduces the re-export, it would reintroduce the cycle.
 *
 * The bundle is a build artifact, so the test builds it on demand and
 * skips itself when esbuild can't be resolved (e.g. during a partial
 * install). It does not need a TTY.
 */

import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { beforeAll, describe, expect, it } from 'vitest'

const here = dirname(fileURLToPath(import.meta.url))
const uiTuiRoot = resolve(here, '..', '..')
const bundlePath = resolve(uiTuiRoot, 'dist', 'entry.js')

function bundleIsFresh(): boolean {
  if (!existsSync(bundlePath)) {
    return false
  }

  try {
    const bundleMtime = statSync(bundlePath).mtimeMs

    const sourceMtime = statSync(resolve(uiTuiRoot, 'packages/hermes-ink/src/entry-exports.ts')).mtimeMs

    return bundleMtime >= sourceMtime
  } catch {
    return false
  }
}

let bundleSrc = ''

beforeAll(() => {
  if (!bundleIsFresh()) {
    // Refresh the bundle so the regression test runs against current
    // sources, not whatever was last committed by hand.
    execFileSync(process.execPath, [resolve(uiTuiRoot, 'scripts/build.mjs')], {
      cwd: uiTuiRoot,
      stdio: ['ignore', 'ignore', 'inherit'],
      timeout: 120_000
    })
  }

  bundleSrc = readFileSync(bundlePath, 'utf8')
}, 180_000)

describe('TUI bundle (issue #31227)', () => {
  it('has no async __esm wrappers (would risk circular-await deadlock)', () => {
    // esbuild emits `async "<path>"() { ... }` as the first key of a
    // module's `__esm` definition when the module body contains
    // top-level await. The lightweight `__esm` helper at the top of
    // the bundle does NOT await nested inits, so any async __esm
    // module in a circular graph hangs forever the first time it's
    // entered.
    const matches = bundleSrc.match(/async "(packages|src|node_modules)\/[^"]+"\s*\(\)/g) ?? []
    expect(
      matches,
      `Found ${matches.length} async __esm wrappers — these can deadlock #31227. First few:\n${matches.slice(0, 3).join('\n')}`
    ).toEqual([])
  })

  it('does not bundle the upstream ink package or ink-text-input', () => {
    // Pulling either of these in re-creates the circular async chain
    // that #31227 was about. The in-tree fork at @hermes/ink replaces
    // all of `ink`; nothing in ui-tui imports `TextInput` from
    // `@hermes/ink` so the re-export is unused dead weight.
    expect(bundleSrc.includes('node_modules/ink/build/index.js')).toBe(false)
    expect(bundleSrc.includes('node_modules/ink-text-input/build/index.js')).toBe(false)
  })

  it('has the @hermes/ink entry-exports module compiled to sync init', () => {
    // Sanity check that the alias swap to packages/hermes-ink/src/entry-exports.ts
    // is still active and producing the expected synchronous init shape.
    expect(bundleSrc).toMatch(/var init_entry_exports = __esm\(\{\s*"packages\/hermes-ink\/src\/entry-exports\.ts"\(\)/)
  })
})
