import { Text, useInput } from 'ink'
import { useEffect, useRef, useState } from 'react'

function wl(s: string, p: number) {
  let i = p - 1
  while (i > 0 && /\s/.test(s[i]!)) i--
  while (i > 0 && !/\s/.test(s[i - 1]!)) i--
  return Math.max(0, i)
}

function wr(s: string, p: number) {
  let i = p
  while (i < s.length && !/\s/.test(s[i]!)) i++
  while (i < s.length && /\s/.test(s[i]!)) i++
  return i
}

const ESC = String.fromCharCode(0x1b)
const INV = ESC + '[7m'
const INV_OFF = ESC + '[27m'
const DIM = ESC + '[2m'
const DIM_OFF = ESC + '[22m'
const PRINTABLE = /^[ -~\u00a0-\uffff]+$/
const BRACKET_PASTE = /\x1b\[20[01]~/g

interface Props {
  value: string
  onChange: (v: string) => void
  onSubmit?: (v: string) => void
  onLargePaste?: (text: string) => string
  placeholder?: string
  focus?: boolean
}

export function TextInput({ value, onChange, onSubmit, onLargePaste, placeholder = '', focus = true }: Props) {
  const [cur, setCur] = useState(value.length)
  const vRef = useRef(value)
  const selfChange = useRef(false)
  const pasteBuf = useRef('')
  const pasteTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pastePos = useRef(0)
  vRef.current = value

  useEffect(() => {
    if (selfChange.current) { selfChange.current = false } else { setCur(value.length) }
  }, [value])

  const flushPaste = () => {
    const pasted = pasteBuf.current
    const at = pastePos.current
    pasteBuf.current = ''
    pasteTimer.current = null
    if (!pasted) return

    const v = vRef.current
    if (pasted.split('\n').length >= 5 || pasted.length > 500) {
      const ph = onLargePaste?.(pasted) ?? pasted.replace(/\n/g, ' ')
      const nv = v.slice(0, at) + ph + v.slice(at)
      selfChange.current = true
      onChange(nv)
      setCur(at + ph.length)
    } else {
      const clean = pasted.replace(/\n/g, ' ')
      if (clean.length && PRINTABLE.test(clean)) {
        const nv = v.slice(0, at) + clean + v.slice(at)
        selfChange.current = true
        onChange(nv)
        setCur(at + clean.length)
      }
    }
  }

  useInput(
    (inp, k) => {
      if (k.upArrow || k.downArrow || (k.ctrl && inp === 'c') || k.tab || (k.shift && k.tab) || k.pageUp || k.pageDown || k.escape)
        return
      if (k.return) { onSubmit?.(value); return }

      let c = cur, v = value
      const mod = k.ctrl || k.meta

      if (k.home || (k.ctrl && inp === 'a'))           c = 0
      else if (k.end || (k.ctrl && inp === 'e'))        c = v.length
      else if (k.leftArrow)                             c = mod ? wl(v, c) : Math.max(0, c - 1)
      else if (k.rightArrow)                            c = mod ? wr(v, c) : Math.min(v.length, c + 1)
      else if ((k.backspace || k.delete) && c > 0) {
        if (mod) { const t = wl(v, c); v = v.slice(0, t) + v.slice(c); c = t }
        else { v = v.slice(0, c - 1) + v.slice(c); c-- }
      }
      else if (k.ctrl && inp === 'w' && c > 0)         { const t = wl(v, c); v = v.slice(0, t) + v.slice(c); c = t }
      else if (k.ctrl && inp === 'u')                   { v = v.slice(c); c = 0 }
      else if (k.ctrl && inp === 'k')                   v = v.slice(0, c)
      else if (k.meta && inp === 'b')                   c = wl(v, c)
      else if (k.meta && inp === 'f')                   c = wr(v, c)
      else if (inp.length > 0) {
        const raw = inp.replace(BRACKET_PASTE, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n')
        if (!raw) return

        const isMultiChar = raw.length > 1 || raw.includes('\n')

        if (isMultiChar) {
          if (!pasteBuf.current) pastePos.current = c
          pasteBuf.current += raw
          if (pasteTimer.current) clearTimeout(pasteTimer.current)
          pasteTimer.current = setTimeout(flushPaste, 50)
          return
        }

        if (PRINTABLE.test(raw)) { v = v.slice(0, c) + raw + v.slice(c); c += raw.length }
        else return
      }
      else return

      c = Math.max(0, Math.min(c, v.length))
      setCur(c)
      if (v !== value) { selfChange.current = true; onChange(v) }
    },
    { isActive: focus }
  )

  if (!focus) return <Text>{value || (placeholder ? DIM + placeholder + DIM_OFF : '')}</Text>
  if (!value && placeholder) return <Text>{INV + (placeholder[0] ?? ' ') + INV_OFF + DIM + placeholder.slice(1) + DIM_OFF}</Text>

  let r = ''
  for (let i = 0; i < value.length; i++) r += i === cur ? INV + value[i] + INV_OFF : value[i]
  if (cur === value.length) r += INV + ' ' + INV_OFF
  return <Text>{r}</Text>
}
