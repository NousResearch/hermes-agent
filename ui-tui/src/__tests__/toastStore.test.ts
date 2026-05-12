import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  dismissToast,
  getToastState,
  patchToastState,
  pushToast,
  resetToastState
} from '../app/toastStore.js'

describe('toastStore', () => {
  beforeEach(() => {
    resetToastState()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('starts empty', () => {
    expect(getToastState().toasts).toEqual([])
  })

  it('adds a toast', () => {
    pushToast('copy', 'Copied to clipboard')

    expect(getToastState().toasts).toHaveLength(1)
    expect(getToastState().toasts[0]!.label).toBe('copy')
    expect(getToastState().toasts[0]!.message).toBe('Copied to clipboard')
    expect(getToastState().toasts[0]!.tone).toBe('info')
  })

  it('auto-dismisses after default duration', () => {
    pushToast('copy', 'Copied to clipboard')
    expect(getToastState().toasts).toHaveLength(1)

    vi.advanceTimersByTime(3000)
    expect(getToastState().toasts).toHaveLength(0)
  })

  it('honors custom duration', () => {
    pushToast('copy', 'Copied to clipboard', 'info', 5000)

    vi.advanceTimersByTime(3000)
    expect(getToastState().toasts).toHaveLength(1)

    vi.advanceTimersByTime(2000)
    expect(getToastState().toasts).toHaveLength(0)
  })

  it('dedupes by label: replaces existing toast and resets timer', () => {
    pushToast('copy', 'First copy')

    vi.advanceTimersByTime(2000)
    expect(getToastState().toasts).toHaveLength(1)
    expect(getToastState().toasts[0]!.message).toBe('First copy')

    pushToast('copy', 'Second copy')
    expect(getToastState().toasts).toHaveLength(1)
    expect(getToastState().toasts[0]!.message).toBe('Second copy')

    vi.advanceTimersByTime(2000)
    expect(getToastState().toasts).toHaveLength(1)

    vi.advanceTimersByTime(1000)
    expect(getToastState().toasts).toHaveLength(0)
  })

  it('keeps distinct labels as separate toasts', () => {
    pushToast('copy', 'Copied')
    pushToast('save', 'Saved')

    expect(getToastState().toasts).toHaveLength(2)
  })

  it('caps toast count at limit and drops oldest', () => {
    pushToast('a', 'A')
    pushToast('b', 'B')
    pushToast('c', 'C')
    pushToast('d', 'D')
    pushToast('e', 'E')

    expect(getToastState().toasts).toHaveLength(4)
    expect(getToastState().toasts.map(t => t.label)).toEqual(['b', 'c', 'd', 'e'])
  })

  it('manual dismiss removes toast immediately', () => {
    pushToast('copy', 'Copied')
    expect(getToastState().toasts).toHaveLength(1)

    dismissToast(getToastState().toasts[0]!.id)
    expect(getToastState().toasts).toHaveLength(0)
  })

  it('returned dismiss function removes toast', () => {
    const dismiss = pushToast('copy', 'Copied')
    expect(getToastState().toasts).toHaveLength(1)

    dismiss()
    expect(getToastState().toasts).toHaveLength(0)
  })

  it('resets to empty', () => {
    pushToast('a', 'A')
    pushToast('b', 'B')

    resetToastState()
    expect(getToastState().toasts).toEqual([])
  })

  it('supports different tones', () => {
    pushToast('ok', 'Done', 'success')
    pushToast('warn', 'Careful', 'warn')
    pushToast('err', 'Failed', 'error')

    const tones = getToastState().toasts.map(t => t.tone)
    expect(tones).toEqual(['success', 'warn', 'error'])
  })

  it('patchToastState merges partial state', () => {
    patchToastState({ toasts: [{ id: 1, label: 'x', message: 'm', tone: 'info', createdAt: 0 }] })
    expect(getToastState().toasts).toHaveLength(1)
  })
})
