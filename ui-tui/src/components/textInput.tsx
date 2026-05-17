import type { InputEvent, Key } from '@hermes/ink'
import * as Ink from '@hermes/ink'
import { type MutableRefObject, useEffect, useMemo, useRef, useState } from 'react'

import { setInputSelection } from '../app/inputSelectionStore.js'
import { readClipboardText, writeClipboardText } from '../lib/clipboard.js'
import { cursorLayout, offsetFromPosition } from '../lib/inputMetrics.js'
import {
  DEFAULT_VOICE_RECORD_KEY,
  isActionMod,
  isMac,
  isMacActionFallback,
  isVoiceToggleKey,
  type ParsedVoiceRecordKey
} from '../lib/platform.js'

type InkExt = typeof Ink & {
  stringWidth: (s: string) => number
  useCursorAdvance: () => (dx: number, dy?: number) => void
  useDeclaredCursor: (a: { line: number; column: number; active: boolean }) => (el: any) => void
  useStdout: () => { stdout?: NodeJS.WriteStream }
  useTerminalFocus: () => boolean
}

const ink = Ink as unknown as InkExt
const { Box, Text, useStdin, useInput, useStdout, stringWidth, useCursorAdvance, useDeclaredCursor, useTerminalFocus } = ink

const ESC = '\x1b'
const INV = `${ESC}[7m`
const INV_OFF = `${ESC}[27m`
const DIM = `${ESC}[2m`
const DIM_OFF = `${ESC}[22m`
const FWD_DEL_RE = new RegExp(`${ESC}\\[3(?:[~$^]|;)`)
const PRINTABLE = /^[ -~\u00a0-\uffff]+$/
const BRACKET_PASTE = new RegExp(`${ESC}?\\[20[01]~`, 'g')
const MULTI_CLICK_MS = 500

const invert = (s: string) => INV + s + INV_OFF
const dim = (s: string) => DIM + s + DIM_OFF

let _seg: Intl.Segmenter | null = null
const seg = () => (_seg ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' }))
const STOP_CACHE_MAX = 32
const stopCache = new Map<string, number[]>()

function graphemeStops(s: string) {
  const hit = stopCache.get(s)

  if (hit) {
    return hit
  }

  const stops = [0]

  for (const { index } of seg().segment(s)) {
    if (index > 0) {
      stops.push(index)
    }
  }

  if (stops.at(-1) !== s.length) {
    stops.push(s.length)
  }

  stopCache.set(s, stops)

  if (stopCache.size > STOP_CACHE_MAX) {
    const oldest = stopCache.keys().next().value

    if (oldest !== undefined) {
      stopCache.delete(oldest)
    }
  }

  return stops
}

function snapPos(s: string, p: number) {
  const pos = Math.max(0, Math.min(p, s.length))
  let last = 0

  for (const stop of graphemeStops(s)) {
    if (stop > pos) {
      break
    }

    last = stop
  }

  return last
}

function prevPos(s: string, p: number) {
  const pos = snapPos(s, p)
  let prev = 0

  for (const stop of graphemeStops(s)) {
    if (stop >= pos) {
      return prev
    }

    prev = stop
  }

  return prev
}

function nextPos(s: string, p: number) {
  const pos = snapPos(s, p)

  for (const stop of graphemeStops(s)) {
    if (stop > pos) {
      return stop
    }
  }

  return s.length
}

function wordLeft(s: string, p: number) {
  let i = snapPos(s, p) - 1

  while (i > 0 && /\s/.test(s[i]!)) {
    i--
  }

  while (i > 0 && !/\s/.test(s[i - 1]!)) {
    i--
  }

  return Math.max(0, i)
}

function wordRight(s: string, p: number) {
  let i = snapPos(s, p)

  while (i < s.length && !/\s/.test(s[i]!)) {
    i++
  }

  while (i < s.length && /\s/.test(s[i]!)) {
    i++
  }

  return i
}

/**
 * Move cursor one logical line up or down inside `s` while preserving the
 * column offset from the current line's start. Returns `null` when the cursor
 * is already on the first line (up) or last line (down) — callers use that
 * signal to fall through to history cycling instead of eating the arrow key.
 */
export function lineNav(s: string, p: number, dir: -1 | 1): null | number {
  const pos = snapPos(s, p)
  const curStart = s.lastIndexOf('\n', pos - 1) + 1
  const col = pos - curStart

  if (dir < 0) {
    if (curStart === 0) {
      return null
    }

    const prevStart = s.lastIndexOf('\n', curStart - 2) + 1

    return snapPos(s, Math.min(prevStart + col, curStart - 1))
  }

  const nextBreak = s.indexOf('\n', pos)

  if (nextBreak < 0) {
    return null
  }

  const nextEnd = s.indexOf('\n', nextBreak + 1)
  const lineEnd = nextEnd < 0 ? s.length : nextEnd

  return snapPos(s, Math.min(nextBreak + 1 + col, lineEnd))
}

export { offsetFromPosition }

const ASCII_PRINTABLE_RE = /^[\x20-\x7e]+$/

/**
 * Pure shape-only precondition for the fast-echo append path.
 *
 * The fast-echo path bypasses Ink's renderer and writes text directly to
 * stdout, so the stored value, the rendered terminal cells, and the cursor
 * column must all stay in sync without any layout work. We only allow it
 * when the inserted text is pure printable ASCII so that:
 *
 *   - `text.length` matches the number of grapheme clusters (no combining
 *     marks, no surrogate pairs, no precomposed CJK / Latin-Extended
 *     letters that an IME might still be holding open as a composition),
 *   - terminal width is exactly 1 cell per character (no East-Asian wide,
 *     no zero-width, no ambiguous-width fonts),
 *   - input methods (Vietnamese Telex, IME, dead-keys) cannot leak
 *     intermediate composition bytes through the bypass before the final
 *     commit arrives — those always go through the normal Ink render path
 *     and stay layout-accurate (closes #5221, #7443, #17602/#17603).
 *
 * We deliberately do NOT just check `stringWidth(text) === text.length`:
 * Vietnamese precomposed letters like "ề" (U+1EC1) report width 1 and
 * length 1 but are still produced by IME compositions and must not be
 * fast-echoed.
 */
export function canFastAppendShape(
  current: string,
  cursor: number,
  text: string,
  columns: number,
  currentLineWidth: number
): boolean {
  if (cursor !== current.length) {
    return false
  }

  if (current.length === 0) {
    return false
  }

  if (current.includes('\n')) {
    return false
  }

  if (!ASCII_PRINTABLE_RE.test(text)) {
    return false
  }

  return currentLineWidth + text.length < Math.max(1, columns)
}

/**
 * Pure shape-only precondition for the fast-echo backspace path.
 *
 * Same reasoning as canFastAppendShape — only allow the direct
 * "\b \b" stdout shortcut when the deleted grapheme is pure printable
 * ASCII. Anything else (combining marks, IME compositions, wide chars,
 * tabs, ANSI fragments) goes through the normal render path so Ink can
 * recompute cell widths.
 *
 * When `columns` is supplied, ALSO rejects when the physical cursor
 * sits at visual column 0 — i.e., right after a soft-wrap boundary.
 * The "\b \b" sequence cannot move the cursor onto the previous visual
 * row (terminals don't back-step across line wraps), so the physical
 * cursor would stay put while the logical caret moves to the end of
 * the previous visual line, desyncing both Ink's `displayCursor` model
 * and the user-visible position.
 *
 * When `columns` is OMITTED, the wrap-boundary check is skipped
 * entirely and the function reverts to the legacy non-wrap-aware
 * contract — values like `'hello '` will return `true` even though
 * they would be unsafe at a width of 6. Production callers (the
 * composer's `canFastBackspace` helper) always pass `columns`;
 * `columns` is optional only so unit tests of the pre-wrap shape
 * contract can keep calling the helper without threading width
 * through. Do NOT omit it from any new caller that relies on the
 * wrap-boundary protection.
 */
export function canFastBackspaceShape(current: string, cursor: number, columns?: number): boolean {
  if (cursor !== current.length) {
    return false
  }

  if (cursor <= 0) {
    return false
  }

  if (current.includes('\n')) {
    return false
  }

  // If we know the wrap width, reject at the soft-wrap boundary: the
  // caret's visual column is 0, so "\b \b" can't represent the physical
  // move back to the previous visual line.
  if (columns !== undefined && cursorLayout(current, cursor, columns).column === 0) {
    return false
  }

  const removed = current.slice(prevPos(current, cursor), cursor)

  return ASCII_PRINTABLE_RE.test(removed)
}

function renderWithCursor(value: string, cursor: number) {
  const pos = Math.max(0, Math.min(cursor, value.length))

  let out = '',
    done = false

  for (const { segment, index } of seg().segment(value)) {
    if (!done && index >= pos) {
      out += invert(index === pos && segment !== '\n' ? segment : ' ')
      done = true

      if (index === pos && segment !== '\n') {
        continue
      }
    }

    out += segment
  }

  return done ? out : out + invert(' ')
}

function renderWithSelection(value: string, start: number, end: number) {
  if (start >= end) {
    return value
  }

  return value.slice(0, start) + invert(value.slice(start, end) || ' ') + value.slice(end)
}

function useFwdDelete(active: boolean) {
  const ref = useRef(false)
  const { inputEmitter: ee } = useStdin()

  useEffect(() => {
    if (!active) {
      return
    }

    const h = (d: string) => {
      ref.current = FWD_DEL_RE.test(d)
    }

    ee.prependListener('input', h)

    return () => {
      ee.removeListener('input', h)
    }
  }, [active, ee])

  return ref
}

type PasteResult = { cursor: number; value: string } | null

const isPasteResultPromise = (
  value: PasteResult | Promise<PasteResult> | null | undefined
): value is Promise<PasteResult> => !!value && typeof (value as PromiseLike<PasteResult>).then === 'function'

export function TextInput({
  columns = 80,
  value,
  onChange,
  onPaste,
  onSubmit,
  mask,
  mouseApiRef,
  voiceRecordKey = DEFAULT_VOICE_RECORD_KEY,
  placeholder = '',
  focus = true
}: TextInputProps) {
  const [cur, setCur] = useState(value.length)
  const [sel, setSel] = useState<null | { end: number; start: number }>(null)
  const fwdDel = useFwdDelete(focus)
  const termFocus = useTerminalFocus()
  const { stdout } = useStdout()
  const noteCursorAdvance = useCursorAdvance()

  const curRef = useRef(cur)
  const selRef = useRef<null | { end: number; start: number }>(null)
  const vRef = useRef(value)
  const self = useRef(false)
  const pasteBuf = useRef('')
  const pasteEnd = useRef<null | number>(null)
  const pasteTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pastePos = useRef(0)
  const editVersionRef = useRef(0)
  const parentChangeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingParentValue = useRef<string | null>(null)
  const localRenderTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lineWidthRef = useRef(stringWidth(value.includes('\n') ? value.slice(value.lastIndexOf('\n') + 1) : value))
  const mouseAnchorRef = useRef<null | number>(null)
  const lastClickRef = useRef<{ at: number; offset: number }>({ at: 0, offset: -1 })
  const undo = useRef<{ cursor: number; value: string }[]>([])
  const redo = useRef<{ cursor: number; value: string }[]>([])

  const cbChange = useRef(onChange)
  const cbSubmit = useRef(onSubmit)
  const cbPaste = useRef(onPaste)
  cbChange.current = onChange
  cbSubmit.current = onSubmit
  cbPaste.current = onPaste

  const raw = self.current ? vRef.current : value
  const display = mask ? raw.replace(/[^\n]/g, mask[0] ?? '*') : raw

  const selected = useMemo(
    () =>
      sel && sel.start !== sel.end ? { end: Math.max(sel.start, sel.end), start: Math.min(sel.start, sel.end) } : null,
    [sel]
  )

  // Read `curRef.current` (always up-to-date) rather than the `cur`
  // React state. The fast-echo path defers the React `setCur` by 16ms
  // to batch re-renders during heavy typing; if an unrelated render
  // flushes this component during that window and we used the stale
  // `cur` state here, the layout effect inside `useDeclaredCursor`
  // would publish a stale cursor declaration and clobber the Ink-level
  // bump from `noteCursorAdvance(...)`. `cur` is still in scope and
  // referenced by setSel/setCur paths below, so React tracks the
  // dependency naturally — we just don't use it as the source of truth
  // for layout. The cursorLayout call is cheap (one wrap-text pass
  // over a single-line string in the common case), so dropping useMemo
  // is fine.
  const layout = cursorLayout(display, curRef.current, columns)

  const boxRef = useDeclaredCursor({
    line: layout.line,
    column: layout.column,
    active: focus && termFocus && !selected
  })

  // Hide the hardware cursor while a selection is active (prevents
  // auto-wrap onto the next row when inverted text fills the column
  // exactly) or when the terminal loses focus (suppresses the hollow-rect
  // ghost most terminals draw at the parked position).
  const hideHardwareCursor = focus && !!stdout?.isTTY && (!!selected || !termFocus)

  useEffect(() => {
    if (!hideHardwareCursor || !stdout) {
      return
    }

    stdout.write('\x1b[?25l')

    return () => {
      stdout.write('\x1b[?25h')
    }
  }, [hideHardwareCursor, stdout])

  const nativeCursor = focus && termFocus && !selected && !!stdout?.isTTY

  // Placeholder text is just a hint, not a selection — render it dim
  // without inverse styling. In a TTY the hardware cursor parks at column
  // 0 and visually marks the input start. Non-TTY surfaces still need the
  // synthetic inverse first-char to draw a cursor at all.
  const rendered = useMemo(() => {
    if (!focus) {
      return display || dim(placeholder)
    }

    if (!display && placeholder) {
      return nativeCursor ? dim(placeholder) : invert(placeholder[0] ?? ' ') + dim(placeholder.slice(1))
    }

    if (selected) {
      return renderWithSelection(display, selected.start, selected.end)
    }

    return nativeCursor ? display || ' ' : renderWithCursor(display, cur)
  }, [cur, display, focus, nativeCursor, placeholder, selected])

  useEffect(() => {
    if (self.current) {
      self.current = false
    } else {
      setCur(value.length)
      setSel(null)
      curRef.current = value.length
      selRef.current = null
      vRef.current = value
      lineWidthRef.current = stringWidth(value.includes('\n') ? value.slice(value.lastIndexOf('\n') + 1) : value)
      undo.current = []
      redo.current = []
    }
  }, [value])

  useEffect(() => {
    if (!focus) {
      return
    }

    const dropSel = () => {
      if (!selRef.current) {
        return
      }

      selRef.current = null
      setSel(null)
    }

    setInputSelection({
      clear: dropSel,
      collapseToEnd: () => {
        dropSel()
        setCur(vRef.current.length)
        curRef.current = vRef.current.length
      },
      end: selected?.end ?? curRef.current,
      start: selected?.start ?? curRef.current,
      value: vRef.current
    })

    return () => setInputSelection(null)
  }, [cur, focus, selected])

  useEffect(
    () => () => {
      if (pasteTimer.current) {
        clearTimeout(pasteTimer.current)
      }

      if (parentChangeTimer.current) {
        clearTimeout(parentChangeTimer.current)
      }

      if (localRenderTimer.current) {
        clearTimeout(localRenderTimer.current)
      }
    },
    []
  )

  const flushParentChange = () => {
    if (parentChangeTimer.current) {
      clearTimeout(parentChangeTimer.current)
      parentChangeTimer.current = null
    }

    const next = pendingParentValue.current
    pendingParentValue.current = null

    if (next !== null) {
      self.current = true
      cbChange.current(next)
    }
  }

  const scheduleParentChange = (next: string) => {
    pendingParentValue.current = next

    if (parentChangeTimer.current) {
      return
    }

    parentChangeTimer.current = setTimeout(flushParentChange, 16)
  }

  const cancelLocalRender = () => {
    if (localRenderTimer.current) {
      clearTimeout(localRenderTimer.current)
      localRenderTimer.current = null
    }
  }

  const scheduleLocalRender = () => {
    if (localRenderTimer.current) {
      return
    }

    localRenderTimer.current = setTimeout(() => {
      localRenderTimer.current = null
      setCur(curRef.current)
    }, 16)
  }

  const canFastEchoBase = () => focus && termFocus && !selected && !mask && !!stdout?.isTTY

  const canFastAppend = (current: string, cursor: number, text: string) =>
    canFastEchoBase() && canFastAppendShape(current, cursor, text, columns, lineWidthRef.current)

  const canFastBackspace = (current: string, cursor: number) =>
    canFastEchoBase() && canFastBackspaceShape(current, cursor, columns)

  const commit = (
    next: string,
    nextCur: number,
    track = true,
    syncParent = true,
    syncLocal = true,
    nextLineWidth?: number
  ) => {
    const prev = vRef.current
    const c = snapPos(next, nextCur)
    editVersionRef.current += 1

    if (selRef.current) {
      selRef.current = null
      setSel(null)
    }

    if (track && next !== prev) {
      undo.current.push({ cursor: curRef.current, value: prev })

      if (undo.current.length > 200) {
        undo.current.shift()
      }

      redo.current = []
    }

    if (syncLocal) {
      cancelLocalRender()
      setCur(c)
    } else {
      scheduleLocalRender()
    }

    curRef.current = c
    vRef.current = next
    lineWidthRef.current =
      nextLineWidth ?? stringWidth(next.includes('\n') ? next.slice(next.lastIndexOf('\n') + 1) : next)

    if (next !== prev) {
      if (syncParent) {
        flushParentChange()
        self.current = true
        cbChange.current(next)
      } else {
        self.current = true
        scheduleParentChange(next)
      }
    }
  }

  const swap = (from: typeof undo, to: typeof redo) => {
    const entry = from.current.pop()

    if (!entry) {
      return
    }

    to.current.push({ cursor: curRef.current, value: vRef.current })
    commit(entry.value, entry.cursor, false)
  }

  const emitPaste = (e: PasteEvent) => {
    const startVersion = editVersionRef.current
    const h = cbPaste.current?.(e)

    if (isPasteResultPromise(h)) {
      const fallbackText = e.text

      void h
        .then(result => {
          if (result && editVersionRef.current === startVersion) {
            commit(result.value, result.cursor)
          } else if (result && fallbackText && PRINTABLE.test(fallbackText)) {
            // User typed while async paste was in-flight — fall back to raw text insert
            // so the pasted content is not silently lost.
            const cur = curRef.current
            const v = vRef.current
            commit(v.slice(0, cur) + fallbackText + v.slice(cur), cur + fallbackText.length)
          }
        })
        .catch(() => {})

      return true
    }

    if (h) {
      commit(h.value, h.cursor)
    }

    return !!h
  }

  const flushPaste = () => {
    const text = pasteBuf.current
    const at = pastePos.current
    const end = pasteEnd.current ?? at
    pasteBuf.current = ''
    pasteEnd.current = null
    pasteTimer.current = null

    if (!text) {
      return
    }

    if (!emitPaste({ cursor: at, text, value: vRef.current }) && PRINTABLE.test(text)) {
      commit(vRef.current.slice(0, at) + text + vRef.current.slice(end), at + text.length)
    }
  }

  const clearSel = () => {
    if (!selRef.current) {
      return
    }

    selRef.current = null
    setSel(null)
  }

  const selectAll = () => {
    const end = vRef.current.length

    if (!end) {
      return
    }

    const next = { end, start: 0 }
    selRef.current = next
    setSel(next)
    setCur(end)
    curRef.current = end
  }

  const moveCursor = (next: number, extend = false) => {
    const c = snapPos(vRef.current, next)
    const anchor = selRef.current?.start ?? curRef.current

    if (!extend || anchor === c) {
      clearSel()
    } else {
      const nextSel = { end: c, start: anchor }
      selRef.current = nextSel
      setSel(nextSel)
    }

    setCur(c)
    curRef.current = c
  }

  const selRange = () => {
    const range = selRef.current

    return range && range.start !== range.end
      ? { end: Math.max(range.start, range.end), start: Math.min(range.start, range.end) }
      : null
  }

  const ins = (v: string, c: number, s: string) => v.slice(0, c) + s + v.slice(c)

  const pastePlainText = (text: string) => {
    const cleaned = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')

    if (!cleaned) {
      return
    }

    const range = selRange()

    const nextValue = range
      ? vRef.current.slice(0, range.start) + cleaned + vRef.current.slice(range.end)
      : vRef.current.slice(0, curRef.current) + cleaned + vRef.current.slice(curRef.current)

    const nextCursor = range ? range.start + cleaned.length : curRef.current + cleaned.length

    commit(nextValue, nextCursor)
  }

  const startMouseSelection = (next: number) => {
    const c = snapPos(vRef.current, next)

    mouseAnchorRef.current = c
    selRef.current = { end: c, start: c }
    setSel(null)
    setCur(c)
    curRef.current = c
  }

  const dragMouseSelection = (next: number) => {
    if (mouseAnchorRef.current === null) {
      return
    }

    const c = snapPos(vRef.current, next)
    const range = { end: c, start: mouseAnchorRef.current }
    selRef.current = range
    setSel(range.start === range.end ? null : range)
    setCur(c)
    curRef.current = c
  }

  const endMouseSelection = () => {
    mouseAnchorRef.current = null

    const range = selRef.current

    if (range && range.start === range.end) {
      selRef.current = null
      setSel(null)

      return
    }

    const normalized = selRange()

    if (isMac && normalized) {
      void writeClipboardText(vRef.current.slice(normalized.start, normalized.end))
    }
  }

  const offsetAt = (e: { localCol?: number; localRow?: number }) =>
    offsetFromPosition(display, e.localRow ?? 0, e.localCol ?? 0, columns)

  const isMultiClickAt = (offset: number) => {
    const now = Date.now()
    const last = lastClickRef.current
    lastClickRef.current = { at: now, offset }

    return now - last.at < MULTI_CLICK_MS && offset === last.offset
  }

  if (mouseApiRef) {
    mouseApiRef.current = {
      dragAt: (row, col) => dragMouseSelection(offsetFromPosition(display, row, col, columns)),
      end: endMouseSelection,
      startAtBeginning: () => startMouseSelection(0)
    }
  }

  useInput(
    (inp: string, k: Key, event: InputEvent) => {
      const eventRaw = event.keypress.raw

      // Configured voice shortcut wins over composer-level defaults like
      // paste/copy so users who bind voice to ctrl+v / alt+v / cmd+v
      // actually get voice toggled instead of a paste (Copilot round-7
      // follow-up on #19835). The pass-through predicate is a no-op for
      // ordinary typing and plain paste when voice is unbound to 'v'.
      if (shouldPassThroughToGlobalHandler(inp, k, voiceRecordKey)) {
        return
      }

      if (
        eventRaw === '\x1bv' ||
        eventRaw === '\x1bV' ||
        eventRaw === '\x16' ||
        (isMac && isActionMod(k) && inp.toLowerCase() === 'v')
      ) {
        if (cbPaste.current) {
          return void emitPaste({ cursor: curRef.current, hotkey: true, text: '', value: vRef.current })
        }

        if (isMac) {
          void readClipboardText().then(text => {
            if (text) {
              pastePlainText(text)
            }
          })
        }

        return
      }

      if (isMac && isActionMod(k) && inp.toLowerCase() === 'c') {
        const range = selRange()

        if (range) {
          const text = vRef.current.slice(range.start, range.end)

          void writeClipboardText(text)
        }

        return
      }

      if (k.upArrow || k.downArrow) {
        const next = lineNav(vRef.current, curRef.current, k.upArrow ? -1 : 1)

        if (next !== null) {
          moveCursor(next, k.shift)

          return
        }

        return
      }

      if (k.return) {
        if (k.shift || k.ctrl || (isMac ? isActionMod(k) : k.meta)) {
          flushParentChange()
          commit(ins(vRef.current, curRef.current, '\n'), curRef.current + 1)
        } else {
          flushParentChange()
          cbSubmit.current?.(vRef.current)
        }

        return
      }

      let c = curRef.current
      let v = vRef.current
      const mod = isActionMod(k)
      const wordMod = mod || k.meta
      const actionHome = k.home || (!isMac && mod && inp === 'a') || isMacActionFallback(k, inp, 'a')
      const actionEnd = k.end || (mod && inp === 'e') || isMacActionFallback(k, inp, 'e')
      const actionDeleteToStart = (mod && inp === 'u') || isMacActionFallback(k, inp, 'u')
      const actionKillToEnd = (mod && inp === 'k') || isMacActionFallback(k, inp, 'k')
      const actionDeleteWord = (mod && inp === 'w') || isMacActionFallback(k, inp, 'w')
      const range = selRange()
      const delFwd = k.delete || fwdDel.current

      if (mod && inp === 'z') {
        return swap(undo, redo)
      }

      if ((mod && inp === 'y') || (mod && k.shift && inp === 'z')) {
        return swap(redo, undo)
      }

      if (isMac && mod && inp === 'a') {
        return selectAll()
      }

      if (actionHome) {
        c = 0
        moveCursor(c, k.shift)

        return
      } else if (actionEnd) {
        c = v.length
        moveCursor(c, k.shift)

        return
      } else if (k.leftArrow) {
        if (range && !wordMod && !k.shift) {
          clearSel()
          c = range.start
        } else {
          c = wordMod ? wordLeft(v, c) : prevPos(v, c)
        }

        moveCursor(c, k.shift)

        return
      } else if (k.rightArrow) {
        if (range && !wordMod && !k.shift) {
          clearSel()
          c = range.end
        } else {
          c = wordMod ? wordRight(v, c) : nextPos(v, c)
        }

        moveCursor(c, k.shift)

        return
      } else if (wordMod && inp === 'b') {
        clearSel()
        c = wordLeft(v, c)
      } else if (wordMod && inp === 'f') {
        clearSel()
        c = wordRight(v, c)
      } else if (range && (k.backspace || delFwd)) {
        v = v.slice(0, range.start) + v.slice(range.end)
        c = range.start
      } else if (k.backspace && c > 0) {
        if (wordMod) {
          const t = wordLeft(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        } else if (canFastBackspace(v, c)) {
          const t = prevPos(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
          stdout!.write('\b \b')
          // The "\b \b" sequence ends with the cursor one column to the
          // LEFT of where Ink last parked it. Tell Ink so its `displayCursor`
          // (and log-update's relative-move basis on the next frame) stays
          // in sync — otherwise the cursor parks one cell to the right of
          // the caret on the next unrelated re-render.
          noteCursorAdvance(-1)
          commit(v, c, true, false, false, Math.max(0, lineWidthRef.current - 1))

          return
        } else {
          const t = prevPos(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        }
      } else if (delFwd && c < v.length) {
        if (wordMod) {
          const t = wordRight(v, c)
          v = v.slice(0, c) + v.slice(t)
        } else {
          v = v.slice(0, c) + v.slice(nextPos(v, c))
        }
      } else if (actionDeleteWord) {
        if (range) {
          v = v.slice(0, range.start) + v.slice(range.end)
          c = range.start
        } else if (c > 0) {
          clearSel()
          const t = wordLeft(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        } else {
          return
        }
      } else if (actionDeleteToStart) {
        if (range) {
          v = v.slice(0, range.start) + v.slice(range.end)
          c = range.start
        } else {
          v = v.slice(c)
          c = 0
        }
      } else if (actionKillToEnd) {
        if (range) {
          v = v.slice(0, range.start) + v.slice(range.end)
          c = range.start
        } else {
          v = v.slice(0, c)
        }
      } else if (event.keypress.isPasted || inp.length > 0) {
        const bracketed = event.keypress.isPasted || inp.includes('[200~')
        const text = inp.replace(BRACKET_PASTE, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n')

        if (bracketed && emitPaste({ bracketed: true, cursor: c, text, value: v })) {
          return
        }

        if (!text) {
          return
        }

        if (text === '\n') {
          return commit(ins(v, c, '\n'), c + 1)
        }

        if (text.length > 1 || text.includes('\n')) {
          if (!pasteBuf.current) {
            pastePos.current = range ? range.start : c
            pasteEnd.current = range ? range.end : pastePos.current
          }

          pasteBuf.current += text

          if (pasteTimer.current) {
            clearTimeout(pasteTimer.current)
          }

          pasteTimer.current = setTimeout(flushPaste, 50)

          return
        }

        if (PRINTABLE.test(text)) {
          if (range) {
            v = v.slice(0, range.start) + text + v.slice(range.end)
            c = range.start + text.length
          } else {
            const simpleAppend = canFastAppend(v, c, text)

            v = v.slice(0, c) + text + v.slice(c)
            c += text.length

            if (simpleAppend) {
              stdout!.write(text)
              // ASCII-printable text advances the physical cursor by exactly
              // text.length cells (canFastAppendShape rejects non-ASCII,
              // wide chars, newlines). Notify Ink so the cached displayCursor
              // / log-update relative-move basis advances with it; otherwise
              // any unrelated re-render that happens before the 16ms
              // setCur/setParent flush parks the cursor text.length cells
              // too far right (#cursor-drift).
              noteCursorAdvance(text.length)
              commit(v, c, true, false, false, lineWidthRef.current + stringWidth(text))

              return
            }
          }
        } else {
          return
        }
      } else {
        return
      }

      commit(v, c)
    },
    { isActive: focus }
  )

  return (
    <Box
      onClick={(e: MouseEventLite) => {
        if (!focus) {
          return
        }

        e.stopImmediatePropagation?.()
        clearSel()
        const next = offsetAt(e)
        setCur(next)
        curRef.current = next
      }}
      onMouseDown={(e: MouseEventLite) => {
        if (!focus) {
          return
        }

        // Right-click → copy active selection if any, otherwise paste.
        if (e.button === 2) {
          e.stopImmediatePropagation?.()
          const decision = decideRightClickAction(vRef.current, selRange())
          if (decision.action === 'copy') {
            void writeClipboardText(decision.text)

            return
          }
          emitPaste({ cursor: curRef.current, hotkey: true, text: '', value: vRef.current })

          return
        }

        if (e.button !== 0) {
          return
        }

        e.stopImmediatePropagation?.()
        const offset = offsetAt(e)

        if (isMultiClickAt(offset)) {
          mouseAnchorRef.current = null
          selectAll()

          return
        }

        startMouseSelection(offset)
      }}
      onMouseDrag={(e: MouseEventLite) => {
        if (!focus || e.button !== 0 || mouseAnchorRef.current === null) {
          return
        }

        e.stopImmediatePropagation?.()
        dragMouseSelection(offsetAt(e))
      }}
      onMouseUp={(e: MouseEventLite) => {
        e.stopImmediatePropagation?.()
        endMouseSelection()
      }}
      ref={boxRef}
      width={columns}
    >
      <Text wrap="wrap">{rendered}</Text>
    </Box>
  )
}

type MouseEventLite = {
  button?: number
  localCol?: number
  localRow?: number
  stopImmediatePropagation?: () => void
}

export interface PasteEvent {
  bracketed?: boolean
  cursor: number
  hotkey?: boolean
  text: string
  value: string
}

interface TextInputProps {
  columns?: number
  focus?: boolean
  mask?: string
  mouseApiRef?: MutableRefObject<null | TextInputMouseApi>
  onChange: (v: string) => void
  onPaste?: (
    e: PasteEvent
  ) => { cursor: number; value: string } | Promise<{ cursor: number; value: string } | null> | null
  onSubmit?: (v: string) => void
  placeholder?: string
  value: string
  voiceRecordKey?: ParsedVoiceRecordKey
}

export type RightClickDecision =
  | { action: 'copy'; text: string }
  | { action: 'paste' }

/**
 * Decide what right-click should do on the composer:
 *   - non-empty selection → copy that text to the clipboard
 *   - no selection (or empty/collapsed range) → fall through to paste
 *
 * Mirrors terminal-native behavior (xterm, iTerm, gnome-terminal) where
 * right-click pastes only when there is nothing selected to copy.
 *
 * Callers pass the already-normalized range from `selRange()` (start <= end,
 * or null when collapsed), so this helper does not need to re-normalize.
 */
export function decideRightClickAction(
  value: string,
  range: { end: number; start: number } | null
): RightClickDecision {
  if (range && range.end > range.start) {
    const text = value.slice(range.start, range.end)
    if (text) {
      return { action: 'copy', text }
    }
  }
  return { action: 'paste' }
}

export const shouldPassThroughToGlobalHandler = (
  input: string,
  key: Key,
  voiceRecordKey: ParsedVoiceRecordKey = DEFAULT_VOICE_RECORD_KEY
): boolean =>
  (key.ctrl && input === 'c') ||
  (key.ctrl && input === 'x') ||
  key.tab ||
  (key.shift && key.tab) ||
  key.pageUp ||
  key.pageDown ||
  key.escape ||
  isVoiceToggleKey(key, input, voiceRecordKey)

export interface TextInputMouseApi {
  dragAt: (row: number, col: number) => void
  end: () => void
  startAtBeginning: () => void
}
