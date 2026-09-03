import { PASTE_SNIPPET_RE } from '../protocol/paste.js'

/**
 * Reference spans in composer text: a `/skill` invoked or named in prose, an
 * `@file:` / `@url:` / `@session:` ref, and a `[[ Image 1 ]]` / paste token.
 * The same vocabulary the desktop chips, so the two surfaces agree on what a
 * reference is.
 *
 * Concatenating every `text` reproduces the input exactly — styling only, the
 * text is never rewritten. Regexes are built per call: a shared `/g` instance
 * carries `lastIndex` between callers and silently skips the first match in
 * the next string it is handed.
 */
export type ComposerHighlight = { ref: boolean; text: string }

// Leading OR mid-prose. `(?![\w-]*\/)` keeps `/usr/local` from lighting up as
// `/usr`, and requiring a letter keeps `a 3 /4 b` plain. A BARE `/` counts only
// at the very end — that is the user opening the command menu, not prose.
const slashRe = () => /(?<=^|\s)(?:\/[a-zA-Z][\w-]*(?![\w-]*\/)|\/$)/g

// Every `@ref` shape the composer accepts: a typed kind (`@file:src/a.ts`), a
// quoted value with spaces, a bare `@diff` / `@staged`, and the half-typed
// `@fi` the user is still working on. Quoted alternatives come before bare
// `\S+` or a quoted value would end at its first space.
const atRe = () => /(?<=^|\s)@(?:[\w-]+:(?:`[^`\n]*`?|"[^"\n]*"?|'[^'\n]*'?|\S*)|\S*)/g

const tokenRe = () => new RegExp(PASTE_SNIPPET_RE.source, 'g')

type Span = { end: number; start: number }

const matchSpans = (text: string, re: RegExp): Span[] =>
  [...text.matchAll(re)].filter(m => m[0]).map(m => ({ end: (m.index ?? 0) + m[0].length, start: m.index ?? 0 }))

export const splitComposerHighlights = (text: string): ComposerHighlight[] => {
  // Tokens, then @refs, then slashes: on an overlap the earlier kind wins, so
  // a slash inside a quoted ref value stays part of that ref.
  const spans = [...matchSpans(text, tokenRe()), ...matchSpans(text, atRe()), ...matchSpans(text, slashRe())]
    .sort((a, b) => a.start - b.start)
    .reduce<Span[]>((kept, span) => {
      if (!kept.some(prev => span.start < prev.end && span.end > prev.start)) {
        kept.push(span)
      }

      return kept
    }, [])

  const out: ComposerHighlight[] = []
  let last = 0

  for (const span of spans) {
    if (span.start > last) {
      out.push({ ref: false, text: text.slice(last, span.start) })
    }

    out.push({ ref: true, text: text.slice(span.start, span.end) })
    last = span.end
  }

  if (last < text.length || !out.length) {
    out.push({ ref: false, text: text.slice(last) })
  }

  return out
}

/** Per-character "is this cell accented", indexed to match the input string. */
export const highlightMask = (text: string): boolean[] => {
  const mask = new Array<boolean>(text.length).fill(false)
  let offset = 0

  for (const segment of splitComposerHighlights(text)) {
    if (segment.ref) {
      mask.fill(true, offset, offset + segment.text.length)
    }

    offset += segment.text.length
  }

  return mask
}

/**
 * Whether every character that stays on screen keeps the color it had.
 *
 * The fast-echo bypass writes ONLY the new cells, so it may run only when a
 * keystroke leaves the existing ones alone. Typing `]` to close a
 * `[[ token ]]`, or a second `/` demoting `/usr` to a path, recolors text
 * already painted — those have to go through a full Ink repaint instead.
 */
export const highlightsStable = (prev: string, next: string): boolean => {
  const before = highlightMask(prev)
  const after = highlightMask(next)

  return before.slice(0, Math.min(before.length, after.length)).every((on, i) => on === after[i])
}
