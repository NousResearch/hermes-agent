import { Text, useInput } from 'ink'
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

const ESC = '\x1b'
const INV = ESC + '[7m'
const INV_OFF = ESC + '[27m'
const DIM = ESC + '[2m'
const DIM_OFF = ESC + '[22m'
const PRINTABLE = /^[ -~\u00a0-\uffff]+$/
const BRACKET_PASTE = new RegExp(`${ESC}\\[20[01]~`, 'g')

interface Props {
  value: string
  onChange: (v: string) => void
  onSubmit?: (v: string) => void
  onPaste?: (data: { cursor: number; text: string; value: string }) => { cursor: number; value: string } | null
  placeholder?: string
  focus?: boolean
}

export function TextInput({ value, onChange, onPaste, onSubmit, placeholder = '', focus = true }: Props) {
  const [cur, setCur] = useState(value.length)

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

  // ── Buffer ops (synchronous, ref-based — no stale closures) ─────

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

  const flushPaste = () => {
    const pasted = pasteBuf.current
    const at = pastePos.current
    pasteBuf.current = ''
    pasteTimer.current = null

    if (!pasted) {
      return
    }

    const v = vRef.current
    const handled = onPasteRef.current?.({ cursor: at, text: pasted, value: v })

    if (handled) {
      return commit(handled.value, handled.cursor)
    }

    if (PRINTABLE.test(pasted)) {
      commit(v.slice(0, at) + pasted + v.slice(at), at + pasted.length)
    }
  }

  // ── Input handler (reads only from refs) ────────────────────────

  useInput(
    (inp, k) => {
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
        if (k.shift || k.meta) {
          commit(vRef.current.slice(0, curRef.current) + '\n' + vRef.current.slice(curRef.current), curRef.current + 1)
        } else {
          onSubmitRef.current?.(vRef.current)
        }

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
      } else if ((k.backspace || k.delete) && c > 0) {
        if (mod) {
          const t = wordLeft(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        } else {
          v = v.slice(0, c - 1) + v.slice(c)
          c--
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
        const raw = inp.replace(BRACKET_PASTE, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n')

        if (!raw) {
          return
        }

        if (raw === '\n') {
          return commit(v.slice(0, c) + '\n' + v.slice(c), c + 1)
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

  const rendered =
    [...value].map((ch, i) => (i === cur ? INV + ch + INV_OFF : ch)).join('') +
    (cur === value.length ? INV + ' ' + INV_OFF : '')

  return <Text>{rendered}</Text>
}
