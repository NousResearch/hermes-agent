/**
 * Kill ring for the Ink composer — what makes `Ctrl+W` / `Ctrl+U` / `Ctrl+K` recoverable.
 *
 * The classic CLI gets this from prompt_toolkit's default emacs bindings, which put every cut
 * on a ring: `Ctrl+Y` pastes the last cut back and `Alt+Y` cycles to older ones. The TUI has
 * always thrown killed text away. This module is the pure half — the ring, the text diff and
 * the yank/yank-pop decision — so the behaviour can be tested without mounting Ink.
 *
 * Key note: `Ctrl+Y` is already **redo** in the TUI (and stays that way in this change), so the
 * paste-back chord here is `Alt+Y`, which is unbound today. See the PR body for the three
 * options the maintainers own.
 */

/** Pieces of killed text kept for the session. Emacs' default is 60; ten is plenty for a
 *  composer and keeps the ring trivially small to reason about. */
export const KILL_RING_SIZE = 10

export class KillRing {
  private readonly maxItems: number
  private items: string[] = []

  constructor(maxItems: number = KILL_RING_SIZE) {
    this.maxItems = Math.max(1, Math.trunc(maxItems))
  }

  get length(): number {
    return this.items.length
  }

  /** Newest-first snapshot (a copy). */
  get entries(): string[] {
    return [...this.items]
  }

  /** Push cut text. Empty or duplicate-of-newest pushes are ignored, as readline does. */
  push(text: string): boolean {
    if (!text) {
      return false
    }

    if (this.items[0] === text) {
      return false
    }

    this.items.unshift(text)
    this.items = this.items.slice(0, this.maxItems)

    return true
  }

  /** Most recent cut, or '' when nothing has been cut yet. */
  latest(): string {
    return this.items[0] ?? ''
  }

  /** Entry at `index`, or '' when the ring is shorter than that. */
  at(index: number): string {
    return this.items[index] ?? ''
  }

  clear(): void {
    this.items = []
  }
}

/**
 * The text a kill chord removed, taken exactly rather than diffed.
 *
 * A kill always removes one contiguous run next to the caret, so the two cursors identify it
 * without ambiguity: a backward kill (Ctrl+W / Ctrl+U / Cmd+Backspace) is `[after, before)`, a
 * forward kill (Ctrl+K / M-d / Cmd+Delete) starts at the old caret and runs for the number of
 * characters the buffer lost. A diff would guess wrong whenever the removed text shares its
 * first or last character with what surrounds it — `one\ntwo\nthree` → Ctrl+K loses `two\n`, but
 * the shared `t` and `e` make a prefix/suffix diff report `wo\nt`.
 *
 * A region delete is covered too: a forward selection ends up in the forward branch, a backward
 * selection in the backward one.
 */
export function removedText(before: string, after: string, cursorBefore: number, cursorAfter: number): string {
  const removed = before.length - after.length

  if (removed <= 0) {
    return ''
  }

  return cursorAfter < cursorBefore
    ? before.slice(cursorAfter, cursorBefore)
    : before.slice(cursorBefore, cursorBefore + removed)
}

export type YankSpan = { end: number; start: number }

/** Where the last yank landed, so a repeat `Alt+Y` knows what it may replace. */
export interface YankState {
  index: number
  span: YankSpan
}

export interface YankResult {
  cursor: number
  next: YankState
  value: string
}

/**
 * One `Alt+Y` press.
 *
 * First press: insert the newest cut at the cursor. Every press straight after a yank (the
 * cursor still sits at the end of the yanked text and the span still holds exactly what was
 * inserted) replaces that text with the next older cut, wrapping at the end of the ring —
 * readline's `yanking` / `yank-pop` pair.
 *
 * Returns null when the ring is empty.
 */
export function applyYank(opts: {
  cursor: number
  last: null | YankState
  ring: KillRing
  value: string
}): null | YankResult {
  const { cursor, last, ring, value } = opts

  if (!ring.length) {
    return null
  }

  const popping =
    last !== null && cursor === last.span.end && value.slice(last.span.start, last.span.end) === ring.at(last.index)

  if (popping) {
    const index = (last.index + 1) % ring.length
    const text = ring.at(index)
    const nextValue = value.slice(0, last.span.start) + text + value.slice(last.span.end)

    return {
      cursor: last.span.start + text.length,
      next: { index, span: { end: last.span.start + text.length, start: last.span.start } },
      value: nextValue
    }
  }

  const text = ring.latest()
  const nextValue = value.slice(0, cursor) + text + value.slice(cursor)

  return {
    cursor: cursor + text.length,
    next: { index: 0, span: { end: cursor + text.length, start: cursor } },
    value: nextValue
  }
}

/**
 * True when a Backspace / ForwardDelete keystroke should kill to the line boundary rather than
 * delete a single word.
 *
 * Only the *super* bit qualifies. It is tempting to reuse `isActionMod`, but that accepts
 * `key.meta` on macOS — and hermes-ink reports Option as `meta`, so Option+Backspace
 * (delete-word, the macOS standard) would be swallowed. On Linux/Windows `isActionMod` is
 * `key.ctrl`, and Ctrl+Backspace is delete-word there too. `super` is set only by kitty CSI-u /
 * xterm modifyOtherKeys, where it unambiguously means Cmd. Terminals that instead rewrite
 * Cmd+Backspace to Ctrl+U are handled by the `isMacActionFallback` kill-to-start path, not here.
 */
export function isLineKillModifier(key: { ctrl: boolean; meta: boolean; super?: boolean }): boolean {
  return key.super === true
}

export interface KillStrokeKey {
  backspace: boolean
  ctrl: boolean
  delete: boolean
  meta: boolean
  super?: boolean
}

/**
 * Whether a keystroke is a readline *kill* — a chord that throws text away and therefore has to
 * put what it removed on the kill ring, so `Alt+Y` can bring it back.
 *
 * Ctrl+W / Ctrl+U / Ctrl+K and their Cmd equivalents, M-d, and the word/line deletions that share
 * Backspace and Delete. A plain Backspace or Delete over a selection is a region delete, not a
 * kill, and stays out of the ring.
 *
 * `mod` is the platform's action modifier (Ctrl off macOS, Cmd on it) and `alt` whether the key
 * carries a meta bit; `isActionFallback` mirrors the composer's `isMacActionFallback`, which lets
 * terminals that rewrite Cmd+<key> to Ctrl+<key> still reach the same command.
 */
export function isKillStroke(
  key: KillStrokeKey,
  inp: string,
  opts: {
    alt: boolean
    delFwd: boolean
    isActionFallback: (key: KillStrokeKey, inp: string, letter: 'a' | 'd' | 'e' | 'k' | 'u' | 'w') => boolean
    mod: boolean
  }
): boolean {
  const { alt, delFwd, isActionFallback, mod } = opts
  const wordMod = mod || alt
  const lineKill = isLineKillModifier(key)
  const action = (letter: 'k' | 'u' | 'w') => (mod && inp === letter) || isActionFallback(key, inp, letter)

  return (
    action('u') ||
    action('k') ||
    action('w') ||
    (wordMod && inp === 'd') ||
    ((key.backspace || delFwd) && (lineKill || wordMod))
  )
}
