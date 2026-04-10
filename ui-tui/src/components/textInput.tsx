import { Text, useInput, useStdin } from 'ink'
import { useEffect, useRef, useState } from 'react'

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

const FWD_DELETE_RE = /\x1b\[3[~$^]|\x1b\[3;/

function useForwardDeleteRef(isActive: boolean) {
  const ref = useRef(false)
  const { internal_eventEmitter: ee } = useStdin()

  useEffect(() => {
    if (!isActive) return

    const onInput = (data: string) => {
      ref.current = FWD_DELETE_RE.test(data)
    }

    ee.prependListener('input', onInput)

    return () => {
      ee.removeListener('input', onInput)
    }
  }, [isActive, ee])

  return ref
}

const ESC = '\x1b'
const INV = ESC + '[7m'
const INV_OFF = ESC + '[27m'
const DIM = ESC + '[2m'
const DIM_OFF = ESC + '[22m'
const PRINTABLE = /^[ -~\u00a0-\uffff]+$/
const BRACKET_PASTE = new RegExp(`${ESC}?\\[20[01]~`, 'g')

export interface PasteEvent {
  bracketed?: boolean
  cursor: number
  hotkey?: boolean
  text: string
  value: string
}

interface Props {
  value: string
  onChange: (v: string) => void
  onSubmit?: (v: string) => void
  onPaste?: (e: PasteEvent) => { cursor: number; value: string } | null
  placeholder?: string
  focus?: boolean
}

export function TextInput({ value, onChange, onPaste, onSubmit, placeholder = '', focus = true }: Props) {
  const [cur, setCur] = useState(value.length)
  const isFwdDelete = useForwardDeleteRef(focus)

  const curRef = useRef(cur)
  const vRef = useRef(value)
  const selfChange = useRef(false)
  const pasteBuf = useRef('')
  const pasteTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pastePos = useRef(0)
  const undoStack = useRef<Array<{ cursor: number; value: string }>>([])
  const redoStack = useRef<Array<{ cursor: number; value: string }>>([])

  const onChangeRef = useRef(onChange)
  const onSubmitRef = useRef(onSubmit)
  const onPasteRef = useRef(onPaste)
  onChangeRef.current = onChange
  onSubmitRef.current = onSubmit
  onPasteRef.current = onPaste

  useEffect(() => {
    if (selfChange.current) {
      selfChange.current = false
    } else {
      setCur(value.length)
      curRef.current = value.length
      vRef.current = value
      undoStack.current = []
      redoStack.current = []
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

  // ── Buffer ops (synchronous, ref-based) ─────────────────────────

  const commit = (next: string, nextCur: number, track = true) => {
    const prev = vRef.current
    const c = Math.max(0, Math.min(nextCur, next.length))

    if (track && next !== prev) {
      undoStack.current.push({ cursor: curRef.current, value: prev })

      if (undoStack.current.length > 200) {
        undoStack.current.shift()
      }

      redoStack.current = []
    }

    setCur(c)
    curRef.current = c
    vRef.current = next

    if (next !== prev) {
      selfChange.current = true
      onChangeRef.current(next)
    }
  }

  const swap = (from: typeof undoStack, to: typeof redoStack) => {
    const entry = from.current.pop()

    if (!entry) {
      return
    }

    to.current.push({ cursor: curRef.current, value: vRef.current })
    commit(entry.value, entry.cursor, false)
  }

  const emitPaste = (e: PasteEvent) => {
    const handled = onPasteRef.current?.(e)

    if (handled) {
      commit(handled.value, handled.cursor)
    }

    return !!handled
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

  const insert = (v: string, c: number, s: string) => v.slice(0, c) + s + v.slice(c)

  // ── Input handler ───────────────────────────────────────────────

  useInput(
    (inp, k) => {
      // Paste hotkeys — single owner, no competing listeners in App
      if ((k.ctrl || k.meta) && inp.toLowerCase() === 'v') {
        emitPaste({ cursor: curRef.current, hotkey: true, text: '', value: vRef.current })

        return
      }

      // Keys handled by App.useInput
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
          ? commit(insert(vRef.current, curRef.current, '\n'), curRef.current + 1)
          : onSubmitRef.current?.(vRef.current)

        return
      }

      let c = curRef.current
      let v = vRef.current
      const mod = k.ctrl || k.meta

      if (k.ctrl && inp === 'z') {
        return swap(undoStack, redoStack)
      }

      if ((k.ctrl && inp === 'y') || (k.meta && k.shift && inp === 'z')) {
        return swap(redoStack, undoStack)
      }

      if (k.home || (k.ctrl && inp === 'a')) {
        c = 0
      } else if (k.end || (k.ctrl && inp === 'e')) {
        c = v.length
      } else if (k.leftArrow) {
        c = mod ? wordLeft(v, c) : Math.max(0, c - 1)
      } else if (k.rightArrow) {
        c = mod ? wordRight(v, c) : Math.min(v.length, c + 1)
      } else if ((k.backspace || k.delete) && !isFwdDelete.current && c > 0) {
        if (mod) {
          const t = wordLeft(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        } else {
          v = v.slice(0, c - 1) + v.slice(c)
          c--
        }
      } else if (k.delete && isFwdDelete.current && c < v.length) {
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
      } else if (k.meta && inp === 'b') {
        c = wordLeft(v, c)
      } else if (k.meta && inp === 'f') {
        c = wordRight(v, c)
      } else if (inp.length > 0) {
        const bracketed = inp.includes('[200~')
        const raw = inp.replace(BRACKET_PASTE, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n')

        if (bracketed && emitPaste({ bracketed: true, cursor: c, text: raw, value: v })) {
          return
        }

        if (!raw) {
          return
        }

        if (raw === '\n') {
          return commit(insert(v, c, '\n'), c + 1)
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

  // ── Render ──────────────────────────────────────────────────────

  if (!focus) {
    return <Text>{value || (placeholder ? DIM + placeholder + DIM_OFF : '')}</Text>
  }

  if (!value && placeholder) {
    return <Text>{INV + (placeholder[0] ?? ' ') + INV_OFF + DIM + placeholder.slice(1) + DIM_OFF}</Text>
  }

  return (
    <Text>
      {[...value].map((ch, i) => (i === cur ? INV + ch + INV_OFF : ch)).join('') +
        (cur === value.length ? INV + ' ' + INV_OFF : '')}
    </Text>
  )
}
