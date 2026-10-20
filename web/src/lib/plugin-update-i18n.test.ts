import { afterEach, expect, it, vi } from 'vitest'

import { formatTranslation, resolveTranslations } from '@/i18n/runtime'
import { api } from '@/lib/api'
import { runDashboardPluginUpdate } from '@/lib/plugin-update'

const pp = resolveTranslations('zh').pluginsPage
const copy = {
  consentBody: (name: string, sha: string) => formatTranslation(pp.updateConsentBody, { name, sha }),
  scanBody: (name: string, count: number) => formatTranslation(pp.updateScanBody, { name, count }),
  scanBlocked: (name: string) => formatTranslation(pp.updateScanBlocked, { name })
}

afterEach(() => vi.restoreAllMocks())

it('shows translated scan consent and never overrides a dangerous verdict', async () => {
  const update = vi
    .spyOn(api, 'updateAgentPlugin')
    .mockResolvedValueOnce({ ok: false, scan_blocked: true, scan_verdict: 'caution', scan_findings: [] })
    .mockResolvedValueOnce({ ok: true })
  const ask = vi.fn().mockReturnValue(true)
  expect(await runDashboardPluginUpdate('plugin $&', copy, ask)).toBe(true)
  expect(ask).toHaveBeenCalledWith('安全扫描在 plugin $& 的新代码中发现了 0 项风险。请仅在信任来源时继续更新：')
  expect(update).toHaveBeenLastCalledWith('plugin $&', { acceptCapabilities: false, force: true })
  ask.mockClear()
  update.mockResolvedValue({ ok: false, scan_blocked: true, scan_verdict: 'dangerous' })
  await expect(runDashboardPluginUpdate('plugin $&', copy, ask)).rejects.toThrow(
    '安全扫描判定 plugin $& 的新代码存在危险，未应用更新。'
  )
  expect(ask).not.toHaveBeenCalled()
})
