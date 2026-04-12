import * as Ink from '@hermes/ink'
import type { InputEvent, Key } from '@hermes/ink'
import { useEffect, useMemo, useRef, useState } from 'react'

type InkExt = typeof Ink & {
  stringWidth: (s: string) => number
  useDeclaredCursor: (a: { line: number; column: number; active: boolean }) => (el: any) => void
  useTerminalFocus: () => boolean
}

const ink = Ink as unknown as InkExt
const { Box, Text, useInput, stringWidth, useDeclaredCursor, useTerminalFocus } = ink

// ── ANSI escapes ─────────────────────────────────────────────────────

const ESC = '\x1b'
const INV = `${ESC}[7m`
const INV_OFF = `${ESC}[27m`
const DIM = `${ESC}[2m`
const DIM_OFF = `${ESC}[22m`
const PRINTABLE = /^[ -~\u00a0-\uffff]+$/
const BRACKET_PASTE = new RegExp(`${ESC}?\\[20[01]~`, 'g')

const invert = (s: string) => INV + s + INV_OFF
const dim = (s: string) => DIM + s + DIM_OFF

// ── Grapheme segmenter (lazy singleton) ──────────────────────────────

let _seg: Intl.Segmenter | null = null
const seg = () => (_seg ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' }))

// ── Word movement ────────────────────────────────────────────────────

function wordLeft(s: string, p: number) {
  let i = p - 1

  while (i > 0 && /\s/.test(s[i]!)) {
    i--
  }

  while (i > 0 && !/\s/.test(s[i - 1]!)) {
    i--
  }

  return Math.max(0, i)
}

function wordRight(s: string, p: number) {
  let i = p

  while (i < s.length && !/\s/.test(s[i]!)) {
    i++
  }

  while (i < s.length && /\s/.test(s[i]!)) {
    i++
  }

  return i
}

// ── Cursor layout (line/column from offset + terminal width) ─────────

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

// ── Render value with inverse-video cursor ───────────────────────────

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

// ── Types ────────────────────────────────────────────────────────────

export interface PasteEvent {
  bracketed?: boolean
  cursor: number
  hotkey?: boolean
  text: string
  value: string
}

interface Props {
  columns?: number
  value: string
  onChange: (v: string) => void
  onSubmit?: (v: string) => void
  onPaste?: (e: PasteEvent) => { cursor: number; value: string } | null
  placeholder?: string
  focus?: boolean
}

// ── Component ────────────────────────────────────────────────────────

export function TextInput({ columns = 80, value, onChange, onPaste, onSubmit, placeholder = '', focus = true }: Props) {
  const [cur, setCur] = useState(value.length)
  const termFocus = useTerminalFocus()

  const curRef = useRef(cur)
  const vRef = useRef(value)
  const self = useRef(false)
  const pasteBuf = useRef('')
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

  const display = self.current ? vRef.current : value

  // ── Cursor declaration ───────────────────────────────────────────

  const layout = useMemo(() => cursorLayout(display, cur, columns), [columns, cur, display])

  const boxRef = useDeclaredCursor({
    line: layout.line,
    column: layout.column,
    active: focus && termFocus
  })

  const rendered = useMemo(() => {
    if (!focus) {
      return display || dim(placeholder)
    }

    if (!display && placeholder) {
      return invert(placeholder[0] ?? ' ') + dim(placeholder.slice(1))
    }

    return renderWithCursor(display, cur)
  }, [cur, display, focus, placeholder])

  // ── Sync external value changes ──────────────────────────────────

  useEffect(() => {
    if (self.current) {
      self.current = false
    } else {
      setCur(value.length)
      curRef.current = value.length
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

  // ── Buffer ops (synchronous, ref-based) ──────────────────────────

  const commit = (next: string, nextCur: number, track = true) => {
    const prev = vRef.current
    const c = Math.max(0, Math.min(nextCur, next.length))

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
    pasteBuf.current = ''
    pasteTimer.current = null

    if (!text) {
      return
    }

    if (!emitPaste({ cursor: at, text, value: vRef.current }) && PRINTABLE.test(text)) {
      commit(vRef.current.slice(0, at) + text + vRef.current.slice(at), at + text.length)
    }
  }

  const ins = (v: string, c: number, s: string) => v.slice(0, c) + s + v.slice(c)

  // ── Input handler ────────────────────────────────────────────────

  useInput(
    (inp: string, k: Key, event: InputEvent) => {
      // Some terminals normalize Ctrl+V to "v"; others deliver raw ^V (\x16).
      const ctrlPaste = k.ctrl && (inp.toLowerCase() === 'v' || event.keypress.raw === '\x16')
      const metaPaste = k.meta && inp.toLowerCase() === 'v'

      if (ctrlPaste || metaPaste) {
        return void emitPaste({ cursor: curRef.current, hotkey: true, text: '', value: vRef.current })
      }

      // Delegated to App
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

      // Undo / redo
      if (k.ctrl && inp === 'z') {
        return swap(undo, redo)
      }

      if ((k.ctrl && inp === 'y') || (k.meta && k.shift && inp === 'z')) {
        return swap(redo, undo)
      }

      // Navigation
      if (k.home || (k.ctrl && inp === 'a')) {
        c = 0
      } else if (k.end || (k.ctrl && inp === 'e')) {
        c = v.length
      } else if (k.leftArrow) {
        c = mod ? wordLeft(v, c) : Math.max(0, c - 1)
      } else if (k.rightArrow) {
        c = mod ? wordRight(v, c) : Math.min(v.length, c + 1)
      } else if (k.meta && inp === 'b') {
        c = wordLeft(v, c)
      } else if (k.meta && inp === 'f') {
        c = wordRight(v, c)
      }

      // Deletion
      else if (k.backspace && c > 0) {
        if (mod) {
          const t = wordLeft(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        } else {
          v = v.slice(0, c - 1) + v.slice(c)
          c--
        }
      } else if (k.delete && c < v.length) {
        if (mod) {
          const t = wordRight(v, c)
          v = v.slice(0, c) + v.slice(t)
        } else {
          v = v.slice(0, c) + v.slice(c + 1)
        }
      } else if (k.ctrl && inp === 'w' && c > 0) {
        const t = wordLeft(v, c)
        v = v.slice(0, t) + v.slice(c)
        c = t
      } else if (k.ctrl && inp === 'u') {
        v = v.slice(c)
        c = 0
      } else if (k.ctrl && inp === 'k') {
        v = v.slice(0, c)
      }

      // Text insertion / paste buffering
      else if (inp.length > 0) {
        const bracketed = inp.includes('[200~')
        const raw = inp.replace(BRACKET_PASTE, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n')

        if (bracketed && emitPaste({ bracketed: true, cursor: c, text: raw, value: v })) {
          return
        }

        if (!raw) {
          return
        }

        if (raw === '\n') {
          return commit(ins(v, c, '\n'), c + 1)
        }

        if (raw.length > 1 || raw.includes('\n')) {
          if (!pasteBuf.current) {
            pastePos.current = c
          }

          pasteBuf.current += raw

          if (pasteTimer.current) {
            clearTimeout(pasteTimer.current)
          }

          pasteTimer.current = setTimeout(flushPaste, 50)

          return
        }

        if (PRINTABLE.test(raw)) {
          v = v.slice(0, c) + raw + v.slice(c)
          c += raw.length
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

  // ── Render ───────────────────────────────────────────────────────

  return (
    <Box ref={boxRef}>
      <Text wrap="wrap">{rendered}</Text>
    </Box>
  )
}
