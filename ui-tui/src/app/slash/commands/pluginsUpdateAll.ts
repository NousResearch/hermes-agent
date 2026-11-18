// `/plugins update` in the TUI: the batch runs over `plugins.manage update_all` (the RPC Desktop uses),
// never inside `slash.exec`, whose worker is killed after 45 s while a clone, scan or dependency sync
// may still be publishing.
import { t } from '../../../i18n/runtime.js'
import type { SlashCommand } from '../types.js'

type Ctx = Parameters<SlashCommand['run']>[1]

interface UpdateAllRow {
  detail?: string
  name: string
  status: 'current' | 'failed' | 'needs_review' | 'updated'
}

export interface PluginsUpdateAllResponse {
  current?: number
  results?: UpdateAllRow[]
  skipped?: { name: string; reason: string }[]
}

// One minute per plugin is the single-update budget; a whole sweep gets ten (as Desktop).
export const UPDATE_ALL_TIMEOUT_MS = 600_000

const STATUS_MARK = {
  current: () => t('slashCmd.ops.pluginsUpdate.current'),
  failed: () => t('slashCmd.ops.pluginsUpdate.failed'),
  needs_review: () => t('slashCmd.ops.pluginsUpdate.needsReview'),
  updated: () => t('slashCmd.ops.pluginsUpdate.updated')
} satisfies Record<UpdateAllRow['status'], () => string>

export function pluginsUpdateAllLines(outcome: PluginsUpdateAllResponse): string[] {
  const results = outcome.results ?? []

  const lines = [
    t('slashCmd.ops.pluginsUpdate.title'),
    ...results.map(row => `  ${STATUS_MARK[row.status]?.() ?? row.status}  ${row.name}${row.detail ? `  (${row.detail})` : ''}`),
    ...(outcome.skipped ?? []).map(skip => `  - ${t('slashCmd.ops.pluginsUpdate.skipped', skip.name, skip.reason)}`)
  ]

  if (!results.length) {
    lines.push(`  ${t('slashCmd.ops.pluginsUpdate.none')}`)
  }

  if (results.some(row => row.status === 'needs_review')) {
    lines.push(`  ${t('slashCmd.ops.pluginsUpdate.review')}`)
  }

  return lines
}

export function runPluginsUpdateAll(ctx: Ctx): void {
  ctx.transcript.sys(t('slashCmd.ops.pluginsUpdate.running'))
  ctx.gateway.gw
    .request<PluginsUpdateAllResponse>('plugins.manage', { action: 'update_all' }, UPDATE_ALL_TIMEOUT_MS)
    .then(outcome => {
      if (ctx.stale()) {
        return
      }

      ctx.transcript.page(pluginsUpdateAllLines(outcome ?? {}).join('\n'), t('slashCmd.ops.slashWorker.pluginsTitle'))
    })
    .catch(ctx.guardedErr)
}
