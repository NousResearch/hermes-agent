import type { InputEvent, Key } from '@hermes/ink'
import * as Ink from '@hermes/ink'
import { useEffect, useMemo, useRef, useState } from 'react'

import { setInputSelection } from '../app/inputSelectionStore.js'
import { readClipboardText, writeClipboardText } from '../lib/clipboard.js'
import { cursorLayout } from '../lib/inputMetrics.js'
import { isActionMod, isMac, isMacActionFallback } from '../lib/platform.js'

type InkExt = typeof Ink & {
  stringWidth: (s: string) => number
  useDeclaredCursor: (a: { line: number; column: number; active: boolean }) => (el: any) => void
  useStdout: () => { stdout?: NodeJS.WriteStream }
  useTerminalFocus: () => boolean
}

const ink = Ink as unknown as InkExt
const { Box, Text, useStdin, useInput, useStdout, stringWidth, useDeclaredCursor, useTerminalFocus } = ink

const ESC = '\x1b'
const INV = `${ESC}[7m`
const INV_OFF = `${ESC}[27m`
const DIM = `${ESC}[2m`
const DIM_OFF = `${ESC}[22m`
const FWD_DEL_RE = new RegExp(`${ESC}\\[3(?:[~$^]|;)`)
const PRINTABLE = /^[ -~\u00a0-\uffff]+$/
const BRACKET_PASTE = new RegExp(`${ESC}?\\[20[01]~`, 'g')

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

export function offsetFromPosition(value: string, row: number, col: number, cols: number) {
  if (!value.length) {
    return 0
  }

  const targetRow = Math.max(0, Math.floor(row))
  const targetCol = Math.max(0, Math.floor(col))
  const w = Math.max(1, cols)

  let line = 0
  let column = 0
  let lastOffset = 0

  for (const { segment, index } of seg().segment(value)) {
    lastOffset = index

    if (segment === '\n') {
      if (line === targetRow) {
        return index
      }

      line++
      column = 0

      continue
    }

    const sw = Math.max(1, stringWidth(segment))

    if (column + sw > w) {
      if (line === targetRow) {
        return index
      }

      line++
      column = 0
    }

    if (line === targetRow && targetCol <= column + Math.max(0, sw - 1)) {
      return index
    }

    column += sw
  }

  if (targetRow >= line) {
    return value.length
  }

  return lastOffset
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
  placeholder = '',
  focus = true
}: TextInputProps) {
  const [cur, setCur] = useState(value.length)
  const [sel, setSel] = useState<null | { end: number; start: number }>(null)
  const fwdDel = useFwdDelete(focus)
  const termFocus = useTerminalFocus()
  const { stdout } = useStdout()

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

  const layout = useMemo(() => cursorLayout(display, cur, columns), [columns, cur, display])

  const boxRef = useDeclaredCursor({
    line: layout.line,
    column: layout.column,
    active: focus && termFocus && !selected
  })

  const nativeCursor = focus && termFocus && !selected && !!stdout?.isTTY

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

    setInputSelection({
      clear: () => {
        if (selRef.current) {
          selRef.current = null
          setSel(null)
        }
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

  const canFastAppend = (current: string, cursor: number, text: string) => {
    const sw = stringWidth(text)

    return (
      canFastEchoBase() &&
      cursor === current.length &&
      current.length > 0 &&
      !current.includes('\n') &&
      sw === text.length &&
      lineWidthRef.current + sw < Math.max(1, columns)
    )
  }

  const canFastBackspace = (current: string, cursor: number) => {
    if (!canFastEchoBase() || cursor !== current.length || cursor <= 0 || current.includes('\n')) {
      return false
    }

    return stringWidth(current.slice(prevPos(current, cursor), cursor)) === 1
  }

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
    lineWidthRef.current = nextLineWidth ?? stringWidth(next.includes('\n') ? next.slice(next.lastIndexOf('\n') + 1) : next)

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

  useInput(
    (inp: string, k: Key, event: InputEvent) => {
      const eventRaw = event.keypress.raw

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
          clearSel()
          setCur(next)
          curRef.current = next

          return
        }

        return
      }

      // Ctrl+B is the documented voice-recording toggle (see platform.ts →
      // isVoiceToggleKey). Pass it through so the app-level handler in
      // useInputHandlers receives it instead of being swallowed here as
      // either backward-word nav (line below) or a literal 'b' insertion.
      if (
        (k.ctrl && inp === 'c') ||
        (k.ctrl && inp === 'b') ||
        k.tab ||
        (k.shift && k.tab) ||
        k.pageUp ||
        k.pageDown ||
        k.escape
      ) {
        return
      }

      if (k.return) {
        if (k.shift || (isMac ? isActionMod(k) : k.meta)) {
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
        clearSel()
        c = 0
      } else if (actionEnd) {
        clearSel()
        c = v.length
      } else if (k.leftArrow) {
        if (range && !wordMod) {
          clearSel()
          c = range.start
        } else {
          clearSel()
          c = wordMod ? wordLeft(v, c) : prevPos(v, c)
        }
      } else if (k.rightArrow) {
        if (range && !wordMod) {
          clearSel()
          c = range.end
        } else {
          clearSel()
          c = wordMod ? wordRight(v, c) : nextPos(v, c)
        }
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
      } else if (inp.length > 0) {
        const bracketed = inp.includes('[200~')
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
      onClick={(e: { localRow?: number; localCol?: number }) => {
        if (!focus) {
          return
        }

        clearSel()
        const next = offsetFromPosition(display, e.localRow ?? 0, e.localCol ?? 0, columns)
        setCur(next)
        curRef.current = next
      }}
      onMouseDown={(e: { button: number }) => {
        // Right-click to paste: route through the same hotkey path as
        // Alt+V so the composer's clipboard RPC (text or image) handles it.
        if (!focus || e.button !== 2) {
          return
        }

        emitPaste({ cursor: curRef.current, hotkey: true, text: '', value: vRef.current })
      }}
      ref={boxRef}
    >
      <Text wrap="wrap-char">{rendered}</Text>
    </Box>
  )
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
  onChange: (v: string) => void
  onPaste?: (
    e: PasteEvent
  ) => { cursor: number; value: string } | Promise<{ cursor: number; value: string } | null> | null
  onSubmit?: (v: string) => void
  placeholder?: string
  value: string
}
