import type {
  AgentsListResponse,
  BrowserManageResponse,
  ConfigShowResponse,
  CronListResponse,
  PluginsListResponse,
  RollbackActionResponse,
  RollbackListResponse,
  SkillsBrowseResponse,
  SkillsListResponse,
  SlashExecResponse,
  ToolsConfigureResponse,
  ToolsetsListResponse,
  ToolsListResponse,
  ToolsShowResponse
} from '../../../gatewayTypes.js'
import type { PanelSection } from '../../../types.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

const passthroughSlash = (ctx: SlashRunCtx, cmd: string, fallback: string) =>
  ctx.gateway.gw
    .request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: ctx.sid })
    .then(r => {
      if (ctx.stale()) {
        return
      }

      ctx.transcript.sys(r?.warning ? `warning: ${r.warning}\n${r?.output || fallback}` : r?.output || fallback)
    })
    .catch(ctx.guardedErr)

const clip = (s: string, max: number) => (s.length > max ? `${s.slice(0, max)}…` : s)

export const opsCommands: SlashCommand[] = [
  {
    help: 'list or restore checkpoints',
    name: 'rollback',
    run: (arg, ctx) => {
      const [sub, ...rest] = (arg || 'list').split(/\s+/)

      if (!sub || sub === 'list') {
        return ctx.gateway.rpc<RollbackListResponse>('rollback.list', { session_id: ctx.sid }).then(
          ctx.guarded<RollbackListResponse>(r => {
            if (!r.checkpoints?.length) {
              return ctx.transcript.sys('no checkpoints')
            }

            ctx.transcript.panel('Checkpoints', [
              {
                rows: r.checkpoints.map(
                  (c, i) => [`${i + 1} ${c.hash?.slice(0, 8) ?? ''}`, c.message ?? ''] as [string, string]
                )
              }
            ])
          })
        )
      }

      const isRestoreOrDiff = sub === 'restore' || sub === 'diff'
      const hash = isRestoreOrDiff ? rest[0] : sub
      const filePath = (isRestoreOrDiff ? rest.slice(1) : rest).join(' ').trim()
      const method = sub === 'diff' ? 'rollback.diff' : 'rollback.restore'

      ctx.gateway
        .rpc<RollbackActionResponse>(method, {
          hash,
          session_id: ctx.sid,
          ...(sub === 'diff' || !filePath ? {} : { file_path: filePath })
        })
        .then(ctx.guarded<RollbackActionResponse>(r => ctx.transcript.sys(r.rendered || r.diff || r.message || 'done')))
    }
  },

  {
    help: 'manage browser connection',
    name: 'browser',
    run: (arg, ctx) => {
      const [action, url] = (arg || 'status').split(/\s+/)

      ctx.gateway
        .rpc<BrowserManageResponse>('browser.manage', { action, ...(url ? { url } : {}) })
        .then(
          ctx.guarded<BrowserManageResponse>(r =>
            ctx.transcript.sys(r.connected ? `browser: ${r.url}` : 'browser: disconnected')
          )
        )
    }
  },

  {
    help: 'list installed plugins',
    name: 'plugins',
    run: (_arg, ctx) => {
      ctx.gateway.rpc<PluginsListResponse>('plugins.list', {}).then(
        ctx.guarded<PluginsListResponse>(r => {
          if (!r.plugins?.length) {
            return ctx.transcript.sys('no plugins')
          }

          ctx.transcript.panel('Plugins', [
            { items: r.plugins.map(p => `${p.name} v${p.version}${p.enabled ? '' : ' (disabled)'}`) }
          ])
        })
      )
    }
  },

  {
    help: 'list or browse skills',
    name: 'skills',
    run: (arg, ctx, cmd) => {
      const [sub, ...rest] = (arg || '').split(/\s+/).filter(Boolean)

      if (!sub || sub === 'list') {
        return ctx.gateway.rpc<SkillsListResponse>('skills.manage', { action: 'list' }).then(
          ctx.guarded<SkillsListResponse>(r => {
            if (!r.skills || !Object.keys(r.skills).length) {
              return ctx.transcript.sys('no skills installed')
            }

            ctx.transcript.panel(
              'Installed Skills',
              Object.entries(r.skills).map(([title, items]) => ({ items, title }))
            )
          })
        )
      }

      if (sub === 'browse') {
        const pageNumber = parseInt(rest[0] ?? '1', 10) || 1

        return ctx.gateway.rpc<SkillsBrowseResponse>('skills.manage', { action: 'browse', page: pageNumber }).then(
          ctx.guarded<SkillsBrowseResponse>(r => {
            if (!r.items?.length) {
              return ctx.transcript.sys('no skills found in the hub')
            }

            const page = r.page ?? 1
            const totalPages = r.total_pages ?? 1

            const sections: PanelSection[] = [
              {
                rows: r.items.map(s => [s.name ?? '', clip(s.description ?? '', 60)] as [string, string])
              }
            ]

            if (page < totalPages) {
              sections.push({ text: `/skills browse ${page + 1} → next page` })
            }

            if (page > 1) {
              sections.push({ text: `/skills browse ${page - 1} → prev page` })
            }

            ctx.transcript.panel(`Skills Hub (page ${page}/${totalPages}, ${r.total ?? 0} total)`, sections)
          })
        )
      }

      passthroughSlash(ctx, cmd, '/skills: no output')
    }
  },

  {
    aliases: ['tasks'],
    help: 'running agents',
    name: 'agents',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<AgentsListResponse>('agents.list', {})
        .then(
          ctx.guarded<AgentsListResponse>(r => {
            const processes = r.processes ?? []
            const running = processes.filter(p => p.status === 'running')
            const finished = processes.filter(p => p.status !== 'running')
            const sections: PanelSection[] = []

            if (running.length) {
              sections.push({
                rows: running.map(p => [p.session_id.slice(0, 8), p.command ?? '']),
                title: `Running (${running.length})`
              })
            }

            if (finished.length) {
              sections.push({
                rows: finished.map(p => [p.session_id.slice(0, 8), p.command ?? '']),
                title: `Finished (${finished.length})`
              })
            }

            if (!sections.length) {
              sections.push({ text: 'No active processes' })
            }

            ctx.transcript.panel('Agents', sections)
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'list or manage cron jobs',
    name: 'cron',
    run: (arg, ctx, cmd) => {
      if (arg && arg !== 'list') {
        return passthroughSlash(ctx, cmd, '(no output)')
      }

      ctx.gateway
        .rpc<CronListResponse>('cron.manage', { action: 'list' })
        .then(
          ctx.guarded<CronListResponse>(r => {
            const jobs = r.jobs ?? []

            if (!jobs.length) {
              return ctx.transcript.sys('no scheduled jobs')
            }

            ctx.transcript.panel('Cron', [
              {
                rows: jobs.map(
                  j =>
                    [j.name || j.job_id?.slice(0, 12) || '', `${j.schedule ?? ''} · ${j.state ?? 'active'}`] as [
                      string,
                      string
                    ]
                )
              }
            ])
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'show configuration',
    name: 'config',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<ConfigShowResponse>('config.show', {})
        .then(
          ctx.guarded<ConfigShowResponse>(r =>
            ctx.transcript.panel(
              'Config',
              (r.sections ?? []).map(s => ({ rows: s.rows, title: s.title }))
            )
          )
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'list, enable, disable tools',
    name: 'tools',
    run: (arg, ctx) => {
      const [subcommand, ...names] = arg.trim().split(/\s+/).filter(Boolean)

      if (!subcommand) {
        return ctx.gateway
          .rpc<ToolsShowResponse>('tools.show', { session_id: ctx.sid })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            if (!r?.sections?.length) {
              return ctx.transcript.sys('no tools')
            }

            ctx.transcript.panel(
              `Tools${typeof r.total === 'number' ? ` (${r.total})` : ''}`,
              r.sections.map(section => ({
                rows: section.tools.map(tool => [tool.name, tool.description] as [string, string]),
                title: section.name
              }))
            )
          })
          .catch(ctx.guardedErr)
      }

      if (subcommand === 'list') {
        return ctx.gateway
          .rpc<ToolsListResponse>('tools.list', { session_id: ctx.sid })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            if (!r?.toolsets?.length) {
              return ctx.transcript.sys('no tools')
            }

            ctx.transcript.panel(
              'Tools',
              r.toolsets.map(ts => ({
                items: ts.tools,
                title: `${ts.enabled ? '*' : ' '} ${ts.name} [${ts.tool_count} tools]`
              }))
            )
          })
          .catch(ctx.guardedErr)
      }

      if (subcommand === 'disable' || subcommand === 'enable') {
        if (!names.length) {
          ctx.transcript.sys(`usage: /tools ${subcommand} <name> [name ...]`)
          ctx.transcript.sys(`built-in toolset: /tools ${subcommand} web`)
          ctx.transcript.sys(`MCP tool: /tools ${subcommand} github:create_issue`)

          return
        }

        return ctx.gateway
          .rpc<ToolsConfigureResponse>('tools.configure', { action: subcommand, names, session_id: ctx.sid })
          .then(
            ctx.guarded<ToolsConfigureResponse>(r => {
              if (r.info) {
                ctx.session.setSessionStartedAt(Date.now())
                ctx.session.resetVisibleHistory(r.info)
              }

              r.changed?.length &&
                ctx.transcript.sys(`${subcommand === 'disable' ? 'disabled' : 'enabled'}: ${r.changed.join(', ')}`)
              r.unknown?.length && ctx.transcript.sys(`unknown toolsets: ${r.unknown.join(', ')}`)
              r.missing_servers?.length && ctx.transcript.sys(`missing MCP servers: ${r.missing_servers.join(', ')}`)
              r.reset && ctx.transcript.sys('session reset. new tool configuration is active.')
            })
          )
          .catch(ctx.guardedErr)
      }

      ctx.transcript.sys('usage: /tools [list|disable|enable] ...')
    }
  },

  {
    help: 'list toolsets',
    name: 'toolsets',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<ToolsetsListResponse>('toolsets.list', { session_id: ctx.sid })
        .then(
          ctx.guarded<ToolsetsListResponse>(r => {
            if (!r.toolsets?.length) {
              return ctx.transcript.sys('no toolsets')
            }

            ctx.transcript.panel('Toolsets', [
              {
                rows: r.toolsets.map(
                  ts =>
                    [`${ts.enabled ? '(*)' : '   '} ${ts.name}`, `[${ts.tool_count}] ${ts.description}`] as [
                      string,
                      string
                    ]
                )
              }
            ])
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
