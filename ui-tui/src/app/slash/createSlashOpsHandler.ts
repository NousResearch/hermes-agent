import type { ToolsConfigureResponse, ToolsListResponse, ToolsShowResponse } from '../../gatewayTypes.js'
import { rpcErrorMessage } from '../../lib/rpc.js'
import type { PanelSection } from '../../types.js'
import type { SlashHandlerContext } from '../interfaces.js'

import { isStaleSlash } from './isStaleSlash.js'
import type { ParsedSlashCommand } from './shared.js'

export function createSlashOpsHandler(ctx: SlashHandlerContext) {
  const { rpc } = ctx.gateway
  const { resetVisibleHistory, setSessionStartedAt } = ctx.session
  const { panel, sys } = ctx.transcript

  return ({ arg, cmd, flight, name, sid }: OpsSlashCommand) => {
    const stale = () => isStaleSlash(ctx, flight, sid)

    switch (name) {
      case 'rollback': {
        const [sub, ...rest] = (arg || 'list').split(/\s+/)

        if (!sub || sub === 'list') {
          rpc('rollback.list', { session_id: sid }).then((r: any) => {
            if (stale() || !r) {
              return
            }

            if (!r.checkpoints?.length) {
              sys('no checkpoints')

              return
            }

            panel('Checkpoints', [
              {
                rows: r.checkpoints.map(
                  (c: any, i: number) => [`${i + 1} ${c.hash?.slice(0, 8)}`, c.message] as [string, string]
                )
              }
            ])
          })

          return true
        }

        const hash = sub === 'restore' || sub === 'diff' ? rest[0] : sub
        const filePath = (sub === 'restore' || sub === 'diff' ? rest.slice(1) : rest).join(' ').trim()

        rpc(sub === 'diff' ? 'rollback.diff' : 'rollback.restore', {
          session_id: sid,
          hash,
          ...(sub === 'diff' || !filePath ? {} : { file_path: filePath })
        }).then((r: any) => {
          if (stale() || !r) {
            return
          }

          sys(r.rendered || r.diff || r.message || 'done')
        })

        return true
      }

      case 'browser': {
        const [action, ...rest] = (arg || 'status').split(/\s+/)

        rpc('browser.manage', { action, ...(rest[0] ? { url: rest[0] } : {}) }).then((r: any) => {
          if (stale() || !r) {
            return
          }

          sys(r.connected ? `browser: ${r.url}` : 'browser: disconnected')
        })

        return true
      }

      case 'plugins':
        rpc('plugins.list', {}).then((r: any) => {
          if (stale() || !r) {
            return
          }

          if (!r.plugins?.length) {
            sys('no plugins')

            return
          }

          panel('Plugins', [
            {
              items: r.plugins.map((p: any) => `${p.name} v${p.version}${p.enabled ? '' : ' (disabled)'}`)
            }
          ])
        })

        return true
      case 'skills': {
        const [sub, ...rest] = (arg || '').split(/\s+/).filter(Boolean)

        if (!sub || sub === 'list') {
          rpc('skills.manage', { action: 'list' }).then((r: any) => {
            if (stale() || !r) {
              return
            }

            const skills = r.skills as Record<string, string[]> | undefined

            if (!skills || !Object.keys(skills).length) {
              sys('no skills installed')

              return
            }

            panel(
              'Installed Skills',
              Object.entries(skills).map(([title, items]) => ({ items, title }))
            )
          })

          return true
        }

        if (sub === 'browse') {
          const pageNumber = parseInt(rest[0] ?? '1', 10) || 1

          rpc('skills.manage', { action: 'browse', page: pageNumber }).then((r: any) => {
            if (stale() || !r) {
              return
            }

            if (!r.items?.length) {
              sys('no skills found in the hub')

              return
            }

            const sections: PanelSection[] = [
              {
                rows: r.items.map(
                  (s: any) =>
                    [s.name ?? '', (s.description ?? '').slice(0, 60) + (s.description?.length > 60 ? '…' : '')] as [
                      string,
                      string
                    ]
                )
              }
            ]

            if (r.page < r.total_pages) {
              sections.push({ text: `/skills browse ${r.page + 1} → next page` })
            }

            if (r.page > 1) {
              sections.push({ text: `/skills browse ${r.page - 1} → prev page` })
            }

            panel(`Skills Hub (page ${r.page}/${r.total_pages}, ${r.total} total)`, sections)
          })

          return true
        }

        ctx.gateway.gw
          .request('slash.exec', { command: cmd.slice(1), session_id: sid })
          .then((r: any) => {
            if (stale()) {
              return
            }

            sys(
              r?.warning
                ? `warning: ${r.warning}\n${r?.output || '/skills: no output'}`
                : r?.output || '/skills: no output'
            )
          })
          .catch((e: unknown) => {
            if (stale()) {
              return
            }

            sys(`error: ${rpcErrorMessage(e)}`)
          })

        return true
      }

      case 'agents':

      case 'tasks':
        rpc('agents.list', {})
          .then((r: any) => {
            if (stale() || !r) {
              return
            }

            const processes = r.processes ?? []
            const running = processes.filter((p: any) => p.status === 'running')
            const finished = processes.filter((p: any) => p.status !== 'running')
            const sections: PanelSection[] = []

            running.length &&
              sections.push({
                title: `Running (${running.length})`,
                rows: running.map((p: any) => [p.session_id.slice(0, 8), p.command])
              })
            finished.length &&
              sections.push({
                title: `Finished (${finished.length})`,
                rows: finished.map((p: any) => [p.session_id.slice(0, 8), p.command])
              })
            !sections.length && sections.push({ text: 'No active processes' })
            panel('Agents', sections)
          })
          .catch((e: unknown) => {
            if (stale()) {
              return
            }

            sys(`error: ${rpcErrorMessage(e)}`)
          })

        return true

      case 'cron':
        if (!arg || arg === 'list') {
          rpc('cron.manage', { action: 'list' })
            .then((r: any) => {
              if (stale() || !r) {
                return
              }

              const jobs = r.jobs ?? []

              if (!jobs.length) {
                sys('no scheduled jobs')

                return
              }

              panel('Cron', [
                {
                  rows: jobs.map(
                    (j: any) =>
                      [j.name || j.job_id?.slice(0, 12), `${j.schedule} · ${j.state ?? 'active'}`] as [string, string]
                  )
                }
              ])
            })
            .catch((e: unknown) => {
              if (stale()) {
                return
              }

              sys(`error: ${rpcErrorMessage(e)}`)
            })
        } else {
          ctx.gateway.gw
            .request('slash.exec', { command: cmd.slice(1), session_id: sid })
            .then((r: any) => {
              if (stale()) {
                return
              }

              sys(r?.warning ? `warning: ${r.warning}\n${r?.output || '(no output)'}` : r?.output || '(no output)')
            })
            .catch((e: unknown) => {
              if (stale()) {
                return
              }

              sys(`error: ${rpcErrorMessage(e)}`)
            })
        }

        return true

      case 'config':
        rpc('config.show', {})
          .then((r: any) => {
            if (stale() || !r) {
              return
            }

            panel(
              'Config',
              (r.sections ?? []).map((s: any) => ({
                title: s.title,
                rows: s.rows
              }))
            )
          })
          .catch((e: unknown) => {
            if (stale()) {
              return
            }

            sys(`error: ${rpcErrorMessage(e)}`)
          })

        return true
      case 'tools': {
        const [subcommand, ...names] = arg.trim().split(/\s+/).filter(Boolean)

        if (!subcommand) {
          rpc<ToolsShowResponse>('tools.show', { session_id: sid })
            .then(r => {
              if (stale()) {
                return
              }

              if (!r?.sections?.length) {
                sys('no tools')

                return
              }

              panel(
                `Tools${typeof r.total === 'number' ? ` (${r.total})` : ''}`,
                r.sections.map(section => ({
                  title: section.name,
                  rows: section.tools.map(tool => [tool.name, tool.description] as [string, string])
                }))
              )
            })
            .catch((e: unknown) => {
              if (stale()) {
                return
              }

              sys(`error: ${rpcErrorMessage(e)}`)
            })

          return true
        }

        if (subcommand === 'list') {
          rpc<ToolsListResponse>('tools.list', { session_id: sid })
            .then(r => {
              if (stale()) {
                return
              }

              if (!r?.toolsets?.length) {
                sys('no tools')

                return
              }

              panel(
                'Tools',
                r.toolsets.map(ts => ({
                  title: `${ts.enabled ? '*' : ' '} ${ts.name} [${ts.tool_count} tools]`,
                  items: ts.tools
                }))
              )
            })
            .catch((e: unknown) => {
              if (stale()) {
                return
              }

              sys(`error: ${rpcErrorMessage(e)}`)
            })

          return true
        }

        if (subcommand === 'disable' || subcommand === 'enable') {
          if (!names.length) {
            sys(`usage: /tools ${subcommand} <name> [name ...]`)
            sys(`built-in toolset: /tools ${subcommand} web`)
            sys(`MCP tool: /tools ${subcommand} github:create_issue`)

            return true
          }

          rpc<ToolsConfigureResponse>('tools.configure', {
            action: subcommand,
            names,
            session_id: sid
          })
            .then(r => {
              if (stale() || !r) {
                return
              }

              if (r.info) {
                setSessionStartedAt(Date.now())
                resetVisibleHistory(r.info)
              }

              r.changed?.length && sys(`${subcommand === 'disable' ? 'disabled' : 'enabled'}: ${r.changed.join(', ')}`)
              r.unknown?.length && sys(`unknown toolsets: ${r.unknown.join(', ')}`)
              r.missing_servers?.length && sys(`missing MCP servers: ${r.missing_servers.join(', ')}`)
              r.reset && sys('session reset. new tool configuration is active.')
            })
            .catch((e: unknown) => {
              if (stale()) {
                return
              }

              sys(`error: ${rpcErrorMessage(e)}`)
            })

          return true
        }

        sys('usage: /tools [list|disable|enable] ...')

        return true
      }

      case 'toolsets':
        rpc('toolsets.list', { session_id: sid })
          .then((r: any) => {
            if (stale() || !r) {
              return
            }

            if (!r.toolsets?.length) {
              sys('no toolsets')

              return
            }

            panel('Toolsets', [
              {
                rows: r.toolsets.map(
                  (ts: any) =>
                    [`${ts.enabled ? '(*)' : '   '} ${ts.name}`, `[${ts.tool_count}] ${ts.description}`] as [
                      string,
                      string
                    ]
                )
              }
            ])
          })
          .catch((e: unknown) => {
            if (stale()) {
              return
            }

            sys(`error: ${rpcErrorMessage(e)}`)
          })

        return true
    }

    return false
  }
}

interface OpsSlashCommand extends ParsedSlashCommand {
  flight: number
  sid: null | string
}
