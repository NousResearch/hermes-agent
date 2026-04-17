import type { InputEvent, Key } from '@hermes/ink'
import * as Ink from '@hermes/ink'
import { useEffect, useMemo, useRef, useState } from 'react'

type InkExt = typeof Ink & {
  stringWidth: (s: string) => number
  useDeclaredCursor: (a: { line: number; column: number; active: boolean }) => (el: any) => void
  useTerminalFocus: () => boolean
}

const ink = Ink as unknown as InkExt
const { Box, Text, useStdin, useInput, stringWidth, useDeclaredCursor, useTerminalFocus } = ink

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

function cursorLayout(value: string, cursor: number, cols: number) {
  const pos = Math.max(0, Math.min(cursor, value.length))
  const w = Math.max(1, cols - 1)

  let col = 0,
    line = 0

  for (const { segment, index } of seg().segment(value)) {
    if (index >= pos) {
      break
    }

    if (segment === '\n') {
      line++
      col = 0

      continue
    }

    const sw = stringWidth(segment)

    if (!sw) {
      continue
    }

    if (col + sw > w) {
      line++
      col = 0
    }

    col += sw
  }

  return { column: col, line }
}

function offsetFromPosition(value: string, row: number, col: number, cols: number) {
  if (!value.length) {
    return 0
  }

  const targetRow = Math.max(0, Math.floor(row))
  const targetCol = Math.max(0, Math.floor(col))
  const w = Math.max(1, cols - 1)

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

  const curRef = useRef(cur)
  const selRef = useRef<null | { end: number; start: number }>(null)
  const vRef = useRef(value)
  const self = useRef(false)
  const pasteBuf = useRef('')
  const pasteEnd = useRef<null | number>(null)
  const pasteTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pastePos = useRef(0)
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

  const rendered = useMemo(() => {
    if (!focus) {
      return display || dim(placeholder)
    }

    if (!display && placeholder) {
      return invert(placeholder[0] ?? ' ') + dim(placeholder.slice(1))
    }

    if (selected) {
      return renderWithSelection(display, selected.start, selected.end)
    }

    return renderWithCursor(display, cur)
  }, [cur, display, focus, placeholder, selected])

  useEffect(() => {
    if (self.current) {
      self.current = false
    } else {
      setCur(value.length)
      setSel(null)
      curRef.current = value.length
      selRef.current = null
      vRef.current = value
      undo.current = []
      redo.current = []
    }
  }, [value])

  useEffect(
    () => () => {
      if (pasteTimer.current) {
        clearTimeout(pasteTimer.current)
      }
    },
    []
  )

  const commit = (next: string, nextCur: number, track = true) => {
    const prev = vRef.current
    const c = snapPos(next, nextCur)

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

    setCur(c)
    curRef.current = c
    vRef.current = next

    if (next !== prev) {
      self.current = true
      cbChange.current(next)
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
    const h = cbPaste.current?.(e)

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

  useInput(
    (inp: string, k: Key, event: InputEvent) => {
      const eventRaw = event.keypress.raw

      if (eventRaw === '\x1bv' || eventRaw === '\x1bV') {
        return void emitPaste({ cursor: curRef.current, hotkey: true, text: '', value: vRef.current })
      }

      if (
        k.upArrow ||
        k.downArrow ||
        (k.ctrl && inp === 'c') ||
        k.tab ||
        (k.shift && k.tab) ||
        k.pageUp ||
        k.pageDown ||
        k.escape
      ) {
        return
      }

      if (k.return) {
        k.shift || k.meta
          ? commit(ins(vRef.current, curRef.current, '\n'), curRef.current + 1)
          : cbSubmit.current?.(vRef.current)

        return
      }

      let c = curRef.current
      let v = vRef.current
      const mod = k.ctrl || k.meta
      const range = selRange()
      const delFwd = k.delete || fwdDel.current

      if (k.ctrl && inp === 'z') {
        return swap(undo, redo)
      }

      if ((k.ctrl && inp === 'y') || (k.meta && k.shift && inp === 'z')) {
        return swap(redo, undo)
      }

      if (k.ctrl && inp === 'a') {
        return selectAll()
      }

      if (k.home) {
        clearSel()
        c = 0
      } else if (k.end || (k.ctrl && inp === 'e')) {
        clearSel()
        c = v.length
      } else if (k.leftArrow) {
        if (range && !mod) {
          clearSel()
          c = range.start
        } else {
          clearSel()
          c = mod ? wordLeft(v, c) : prevPos(v, c)
        }
      } else if (k.rightArrow) {
        if (range && !mod) {
          clearSel()
          c = range.end
        } else {
          clearSel()
          c = mod ? wordRight(v, c) : nextPos(v, c)
        }
      } else if (k.meta && inp === 'b') {
        clearSel()
        c = wordLeft(v, c)
      } else if (k.meta && inp === 'f') {
        clearSel()
        c = wordRight(v, c)
      } else if (range && (k.backspace || delFwd)) {
        v = v.slice(0, range.start) + v.slice(range.end)
        c = range.start
      } else if (k.backspace && c > 0) {
        if (mod) {
          const t = wordLeft(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        } else {
          const t = prevPos(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        }
      } else if (delFwd && c < v.length) {
        if (mod) {
          const t = wordRight(v, c)
          v = v.slice(0, c) + v.slice(t)
        } else {
          v = v.slice(0, c) + v.slice(nextPos(v, c))
        }
      } else if (k.ctrl && inp === 'w') {
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
      } else if (k.ctrl && inp === 'u') {
        if (range) {
          v = v.slice(0, range.start) + v.slice(range.end)
          c = range.start
        } else {
          v = v.slice(c)
          c = 0
        }
      } else if (k.ctrl && inp === 'k') {
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
            v = v.slice(0, c) + text + v.slice(c)
            c += text.length
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
      ref={boxRef}
    >
      <Text wrap="wrap">{rendered}</Text>
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
  onPaste?: (e: PasteEvent) => { cursor: number; value: string } | null
  onSubmit?: (v: string) => void
  placeholder?: string
  value: string
}
