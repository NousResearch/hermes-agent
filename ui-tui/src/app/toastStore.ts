import { atom } from 'nanostores'

export type ToastTone = 'error' | 'info' | 'success' | 'warn'

export interface ToastItem {
  createdAt: number
  id: number
  label: string
  message: string
  tone: ToastTone
}

interface ToastState {
  toasts: ToastItem[]
}

const buildToastState = (): ToastState => ({ toasts: [] })

export const $toastState = atom<ToastState>(buildToastState())

export const getToastState = () => $toastState.get()

export const patchToastState = (next: Partial<ToastState> | ((state: ToastState) => ToastState)) =>
  $toastState.set(typeof next === 'function' ? next($toastState.get()) : { ...$toastState.get(), ...next })

export const resetToastState = () => $toastState.set(buildToastState())

let toastIdCounter = 0
const timers = new Map<number, ReturnType<typeof setTimeout>>()

const DEFAULT_TOAST_DURATION_MS = 3000
const TOAST_LIMIT = 4

export function dismissToast(id: number) {
  const t = timers.get(id)

  if (t) {
    clearTimeout(t)
    timers.delete(id)
  }

  patchToastState(state => ({ ...state, toasts: state.toasts.filter(item => item.id !== id) }))
}

export function pushToast(
  label: string,
  message: string,
  tone: ToastTone = 'info',
  durationMs = DEFAULT_TOAST_DURATION_MS
): () => void {
  const now = Date.now()
  const existing = getToastState().toasts.find(t => t.label === label)
  const id = existing ? existing.id : ++toastIdCounter

  if (existing) {
    const oldTimer = timers.get(id)

    if (oldTimer) {
      clearTimeout(oldTimer)
    }
  }

  patchToastState(state => {
    const base = state.toasts.filter(t => t.label !== label)
    const next = [...base, { id, label, message, tone, createdAt: now }].slice(-TOAST_LIMIT)

    return { ...state, toasts: next }
  })

  const timer = setTimeout(() => dismissToast(id), durationMs)

  timers.set(id, timer)

  return () => dismissToast(id)
}
