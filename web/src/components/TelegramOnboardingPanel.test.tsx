// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { MessagingPlatform } from '@/lib/api'

const calls = vi.hoisted(() => ({ start: vi.fn(), poll: vi.fn(), apply: vi.fn(), cancel: vi.fn() }))
vi.mock('@/lib/api', () => ({
  HERMES_BASE_PATH: '',
  getManagementProfile: () => 'test',
  api: {
    startTelegramOnboarding: calls.start,
    getTelegramOnboardingStatus: calls.poll,
    applyTelegramOnboarding: calls.apply,
    cancelTelegramOnboarding: calls.cancel
  }
}))
vi.mock('qrcode', () => ({ toDataURL: async () => 'data:image/png;base64,AA==' }))
import { TelegramOnboardingPanel } from './TelegramOnboardingPanel'

let root: Root
let container: HTMLDivElement
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
const button = (text: string) => [...container.querySelectorAll('button')].find(el => el.textContent?.includes(text))!
const click = async (text: string) =>
  act(async () => {
    button(text).click()
  })
const render = async () =>
  act(async () =>
    root.render(
      <TelegramOnboardingPanel
        platform={{ configured: false } as MessagingPlatform}
        onChanged={async () => {}}
        onManualSetup={() => {}}
        onRestartNeeded={() => {}}
        setRestartNeeded={() => {}}
        showToast={() => {}}
      />
    )
  )

beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  sessionStorage.clear()
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
  calls.start.mockResolvedValue({
    pairing_id: 'pair-example',
    suggested_username: 'suggested_bot',
    deep_link: 'https://t.me/Manager?start=pair_example',
    qr_payload: 'https://t.me/Manager?start=pair_example',
    expires_at: new Date(Date.now() + 1000).toISOString()
  })
  calls.poll.mockResolvedValue({
    status: 'ready',
    bot_username: 'renamed_bot',
    owner_user_id: '42',
    expires_at: new Date(Date.now() + 30 * 60 * 1000).toISOString()
  })
  calls.cancel.mockResolvedValue({ ok: true })
})
afterEach(async () => {
  await act(async () => root.unmount())
  container.remove()
  vi.useRealTimers()
})

it('uses the ready deadline and detected numeric owner after a reload, then resets a terminal apply failure', async () => {
  calls.poll.mockRejectedValueOnce(new Error('502: temporary network failure'))
  await render()
  await click('Create with QR')
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500)
  })
  expect(container.textContent).toContain('Still waiting for Telegram')
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000)
  })
  expect(container.textContent).toContain('@renamed_bot')
  expect(container.textContent).toContain('owner detected')
  expect(button('Save and restart').disabled).toBe(false)
  await act(async () => root.unmount())
  root = createRoot(container)
  await render()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500)
  })
  expect(container.textContent).toContain('@renamed_bot')
  calls.apply.mockRejectedValue(new Error('404: Telegram setup session was not found'))
  await click('Save and restart')
  expect(calls.apply).toHaveBeenCalledWith('pair-example', { allowed_user_ids: ['42'] })
  expect(button('Create with QR').disabled).toBe(false)
  expect(button('Manual setup').disabled).toBe(false)
  expect(container.textContent).toContain('Start a new setup')
  expect(sessionStorage.length).toBe(0)
})

it('returns to a fresh setup when ready confirmation expires or a pending session disappears', async () => {
  calls.poll.mockResolvedValueOnce({
    status: 'ready',
    bot_username: 'bot',
    owner_user_id: '42',
    expires_at: new Date(Date.now() + 4000).toISOString()
  })
  await render()
  await click('Create with QR')
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500)
  })
  expect(button('Save and restart')).toBeDefined()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(3500)
  })
  expect(button('Create with QR').disabled).toBe(false)
  expect(container.textContent).toContain('confirmation expired')
  calls.poll.mockRejectedValue(new Error('404: Telegram setup session was not found'))
  await click('Create with QR')
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500)
  })
  expect(button('Create with QR').disabled).toBe(false)
  expect(container.textContent).not.toContain('Still waiting')
})
