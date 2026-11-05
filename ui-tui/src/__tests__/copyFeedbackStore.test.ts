import { afterEach, expect, it, vi } from 'vitest'

import { $copyFeedback, clearCopyFeedback, type CopyDelivery, runCopyWithFeedback } from '../app/copyFeedbackStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

afterEach(() => {
  clearCopyFeedback()
  resetUiState()
  vi.useRealTimers()
})

it('shows feedback while busy, refreshes its lifetime after another copy, and expires', async () => {
  vi.useFakeTimers()
  patchUiState({ busy: true, sid: 'current' })
  await runCopyWithFeedback(async () => 'terminal')
  expect($copyFeedback.get()).toBe('terminal')
  await vi.advanceTimersByTimeAsync(2000)
  await runCopyWithFeedback(async () => 'clipboard')
  await vi.advanceTimersByTimeAsync(400)
  expect($copyFeedback.get()).toBe('clipboard')
  await vi.advanceTimersByTimeAsync(2000)
  expect($copyFeedback.get()).toBeNull()
})

it('does not confirm failed or empty copies, or let an older completion replace a newer copy', async () => {
  let finishOld!: (delivery: CopyDelivery) => void

  const old = runCopyWithFeedback(
    () =>
      new Promise(resolve => {
        finishOld = resolve
      })
  )

  await runCopyWithFeedback(async () => 'clipboard')
  finishOld('terminal')
  await old
  expect($copyFeedback.get()).toBe('clipboard')
  await runCopyWithFeedback(async () => {
    throw new Error('clipboard unavailable')
  })
  expect($copyFeedback.get()).toBeNull()
  await runCopyWithFeedback(async () => 'terminal')
  await runCopyWithFeedback(async () => null)
  expect($copyFeedback.get()).toBeNull()
})

it('discards a pending confirmation after a session change or cleanup', async () => {
  patchUiState({ sid: 'before' })
  let finish!: (delivery: CopyDelivery) => void

  const pending = runCopyWithFeedback(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )

  patchUiState({ sid: 'after' })
  finish('terminal')
  await pending
  expect($copyFeedback.get()).toBeNull()

  const cancelled = runCopyWithFeedback(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )

  clearCopyFeedback()
  finish('clipboard')
  await cancelled
  expect($copyFeedback.get()).toBeNull()
})
