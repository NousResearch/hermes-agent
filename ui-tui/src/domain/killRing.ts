/**
 * Kill ring for the Ink composer — what makes `Ctrl+W` / `Ctrl+U` / `Ctrl+K` recoverable.
 *
 * The classic CLI gets this from prompt_toolkit's emacs bindings: every cut goes on a ring and
 * can be pasted back. `Ctrl+Y` is already redo in the TUI (unchanged here), so the paste-back
 * chord is `Alt+Y`: the first press yanks the newest cut, each press straight after replaces it
 * with the next older one (readline's yank-pop).
 */

const KILL_RING_SIZE = 10

export class KillRing {
  private items: string[] = []

  get length(): number {
    return this.items.length
  }

  /** Newest first. Empty or duplicate-of-newest pushes are ignored, as readline does. */
  push(text: string): void {
    if (text && this.items[0] !== text) {
      this.items = [text, ...this.items].slice(0, KILL_RING_SIZE)
    }
  }

  at(index: number): string {
    return this.items[index] ?? ''
  }
}

/**
 * Whether a keystroke is a readline kill — Ctrl+W / Ctrl+U / Ctrl+K (or their Cmd forms), M-d, or
 * a word/line Backspace or Delete — rather than a plain or selection delete. Only kills feed the
 * ring. `deleting` is "Backspace or forward Delete"; `lineKill` the composer's Cmd+delete test.
 */
export function isKillChord(
  inp: string,
  f: {
    deleteToStart: boolean
    deleteWord: boolean
    deleting: boolean
    killToEnd: boolean
    lineKill: boolean
    wordMod: boolean
  }
): boolean {
  return (
    f.deleteToStart ||
    f.killToEnd ||
    f.deleteWord ||
    (f.wordMod && inp === 'd') ||
    (f.deleting && (f.lineKill || f.wordMod))
  )
}

/**
 * The text a kill removed, taken from the two cursors rather than diffed: a backward kill is
 * `[after, before)`, a forward kill starts at the old caret. A diff guesses wrong when the cut
 * shares edge characters with its surroundings (`one\ntwo\nthree` + Ctrl+K on `two` would read
 * as `wo\nt`).
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

/** Where the last yank landed, so a repeat `Alt+Y` knows what it may replace. */
export interface YankState {
  end: number
  index: number
  start: number
}

/**
 * One `Alt+Y` press: insert the newest cut at the cursor, or — straight after a yank, with the
 * cursor still at its end and its text untouched — swap it for the next older cut, wrapping.
 * Null when nothing has been cut yet.
 */
export function applyYank(
  ring: KillRing,
  value: string,
  cursor: number,
  last: null | YankState
): null | { cursor: number; next: YankState; value: string } {
  if (!ring.length) {
    return null
  }

  const popping = last !== null && cursor === last.end && value.slice(last.start, last.end) === ring.at(last.index)
  const index = popping ? (last.index + 1) % ring.length : 0
  const start = popping ? last.start : cursor
  const end = popping ? last.end : cursor
  const text = ring.at(index)

  return {
    cursor: start + text.length,
    next: { end: start + text.length, index, start },
    value: value.slice(0, start) + text + value.slice(end)
  }
}
