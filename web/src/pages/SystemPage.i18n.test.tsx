// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { I18nContext, formatTranslation, resolveTranslations } from '@/i18n/runtime'
import type { Locale } from '@/i18n/types'

import SystemPage from './SystemPage'

const apiMocks = vi.hoisted(() => ({
  getStatus: vi.fn(),
  getSystemStats: vi.fn(),
  getMemory: vi.fn(),
  getCredentialPool: vi.fn(),
  getCheckpoints: vi.fn(),
  getHooks: vi.fn(),
  getCurator: vi.fn(),
  getPortal: vi.fn(),
  getGatewayMigratePlan: vi.fn(),
  checkHermesUpdate: vi.fn()
}))

vi.mock('@/lib/api', () => ({ api: apiMocks }))
vi.mock('@/components/HermesConsoleModal', () => ({ HermesConsoleModal: () => null }))

let root: Root
let container: HTMLDivElement

beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
  for (const fn of Object.values(apiMocks)) fn.mockReset().mockRejectedValue(new Error('unavailable'))
  apiMocks.getStatus.mockResolvedValue({})
  apiMocks.getCredentialPool.mockResolvedValue({ providers: [] })
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
})

afterEach(async () => {
  await act(async () => root.unmount())
  container.remove()
  vi.unstubAllGlobals()
})

async function render(locale: Locale) {
  await act(async () => {
    root.render(
      <I18nContext.Provider value={{
        format: formatTranslation,
        locale,
        setLocale: async () => {},
        t: resolveTranslations(locale)
      }}>
        <MemoryRouter><SystemPage /></MemoryRouter>
      </I18nContext.Provider>
    )
  })
}

const updateInfo = {
  install_method: 'git',
  current_version: '1.0.0',
  behind: -1,
  update_available: true,
  can_apply: true,
  update_command: 'hermes update',
  message: 'This build is newer than stable release v0.9.0. Updating moves it back to that release.'
}

it.each(['en', 'zh', 'zh-hant'] as const)('distinguishes a stable rollback in the %s badge and forced-check notice', async locale => {
  apiMocks.checkHermesUpdate.mockResolvedValue({ ...updateInfo, rollback_to: '0.9.0' })
  await render(locale)
  const copy = resolveTranslations(locale).systemPage
  expect(container.textContent).toContain(formatTranslation(copy.host.newerThanStable, { version: '0.9.0' }))
  const check = [...container.querySelectorAll('button')].find(button => button.textContent === copy.host.checkUpdates)
  expect(check).toBeDefined()
  await act(async () => check!.click())
  expect(document.body.textContent).toContain(formatTranslation(copy.toast.rollbackAvailable, { version: '0.9.0' }))
  expect(apiMocks.checkHermesUpdate.mock.calls.map(([force]) => force)).toEqual([false, true])
})

it('keeps a forward-update count while a language change refreshes its copy without another request', async () => {
  apiMocks.checkHermesUpdate.mockResolvedValue({ ...updateInfo, behind: 3, message: null })
  await render('en')
  expect(container.textContent).toContain(formatTranslation(resolveTranslations('en').systemPage.host.behind, { count: 3 }))
  await render('zh')
  const copy = resolveTranslations('zh').systemPage
  expect(container.textContent).toContain(formatTranslation(copy.host.behind, { count: 3 }))
  expect(apiMocks.checkHermesUpdate).toHaveBeenCalledTimes(1)
  const check = [...container.querySelectorAll('button')].find(button => button.textContent === copy.host.checkUpdates)
  await act(async () => check!.click())
  expect(document.body.textContent).toContain(formatTranslation(copy.toast.updateBehind, { count: 3 }))
})
