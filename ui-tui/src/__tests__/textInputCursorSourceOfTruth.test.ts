import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

// Locate textInput.tsx relative to this test file so the assertion
// survives moves of the test fixture itself.
const TEXT_INPUT_PATH = join(dirname(fileURLToPath(import.meta.url)), '..', 'components', 'textInput.tsx')
const source = readFileSync(TEXT_INPUT_PATH, 'utf8')

// Closes Copilot follow-up on PR #26717: the original cursor-drift
// fix bumped Ink's displayCursor / cursorDeclaration on fast-echo, but
// if TextInput itself re-renders before the deferred 16ms `setCur`
// flushes (parent state change, status-bar tick, spinner) the layout
// effect inside `useDeclaredCursor` re-publishes a declaration
// computed from the STALE React `cur` state and clobbers the Ink-level
// bump. The fix is structural: read `curRef.current` (always
// up-to-date) when computing the layout, not the `cur` state.
//
// This file pins that invariant. Switching back to `cur` state — or
// re-introducing a memo keyed on `cur` that uses `curRef.current`
// inside but stops re-computing on rerender — is a regression and
// should be caught here, not via a flaky integration test that mounts
// Ink + stdin.
describe('textInput cursor-layout source of truth', () => {
  it('reads curRef.current (not the cur React state) for cursorLayout', () => {
    // The line we care about. We allow whitespace / formatting drift,
    // but the call itself must use `curRef.current`.
    expect(source).toMatch(/cursorLayout\(\s*display\s*,\s*curRef\.current\s*,\s*columns\s*\)/)
  })

  it('does not pass the bare `cur` React state into cursorLayout', () => {
    // Any `cursorLayout(display, cur, columns)` invocation would
    // reintroduce the stale-declaration window.
    expect(source).not.toMatch(/cursorLayout\(\s*display\s*,\s*cur\s*,\s*columns\s*\)/)
  })

  it('keeps the fast-echo notifier calls paired with the stdout writes', () => {
    // Both fast-echo paths must call noteCursorAdvance, otherwise Ink
    // never learns about the out-of-band write and drifts again. We
    // tolerate explanatory comments in between (the rationale block is
    // intentionally long), but the pairing itself must hold.
    const backspacePattern = /stdout!\.write\(['"`]\\b \\b['"`]\)[\s\S]{0,1000}?noteCursorAdvance\(-1\)/
    expect(source).toMatch(backspacePattern)

    const appendPattern = /stdout!\.write\(text\)[\s\S]{0,1000}?noteCursorAdvance\(text\.length\)/
    expect(source).toMatch(appendPattern)
  })
})
