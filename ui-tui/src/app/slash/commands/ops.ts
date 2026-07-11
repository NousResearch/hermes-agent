import type {
  BrowserManageResponse,
  CommandsCatalogResponse,
  DelegationPauseResponse,
  ProcessStopResponse,
  ReloadEnvResponse,
  ReloadMcpResponse,
  RollbackDiffResponse,
  RollbackListResponse,
  RollbackRestoreResponse,
  SlashExecResponse,
  SpawnTreeListResponse,
  SpawnTreeLoadResponse,
  ToolsConfigureResponse
} from '../../../gatewayTypes.js'
import { translate } from '../../../i18n/index.js'
import type { PanelSection } from '../../../types.js'
import { applyDelegationStatus, getDelegationState } from '../../delegationStore.js'
import { patchOverlayState } from '../../overlayStore.js'
import { getSpawnHistory, pushDiskSnapshot, setDiffPair, type SpawnSnapshot } from '../../spawnHistoryStore.js'
import type { SlashCommand } from '../types.js'

interface SkillInfo {
  category?: string
  description?: string
  name?: string
  path?: string
}

interface SkillsListResponse {
  skills?: Record<string, string[]>
}

interface SkillsInspectResponse {
  info?: SkillInfo
}

interface SkillsSearchResponse {
  results?: { description?: string; name: string }[]
}

interface SkillsInstallResponse {
  installed?: boolean
  name?: string
}

interface SkillsBrowseItem {
  description?: string
  name: string
  source?: string
  trust?: string
}

interface SkillsBrowseResponse {
  items?: SkillsBrowseItem[]
  page?: number
  total?: number
  total_pages?: number
}

interface SkillsReloadResponse {
  output?: string
}

export const opsCommands: SlashCommand[] = [
  {
    help: 'stop background processes',
    name: 'stop',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<ProcessStopResponse>('process.stop', {})
        .then(
          ctx.guarded<ProcessStopResponse>(r => {
            const killed = Number(r.killed ?? 0)
            const noun = killed === 1 ? 'process' : 'processes'
            ctx.transcript.sys(translate(ctx.ui.locale, 'sys.stoppedProcesses', { count: String(killed), noun }))
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    aliases: ['reload_mcp'],
    help: 'reload MCP servers in the live session (warns about prompt cache invalidation)',
    name: 'reload-mcp',
    run: (arg, ctx) => {
      // Parse arg: `now` / `always` skip the confirmation gate.
      // `always` additionally persists approvals.mcp_reload_confirm=false.
      const a = (arg || '').trim().toLowerCase()

      const params: { session_id: string | null; confirm?: boolean; always?: boolean } = {
        session_id: ctx.sid
      }

      if (a === 'now' || a === 'approve' || a === 'once' || a === 'yes') {
        params.confirm = true
      } else if (a === 'always') {
        params.confirm = true
        params.always = true
      }

      ctx.gateway
        .rpc<ReloadMcpResponse>('reload.mcp', params)
        .then(
          ctx.guarded<ReloadMcpResponse>(r => {
            if (r.status === 'confirm_required') {
              ctx.transcript.sys(r.message || translate(ctx.ui.locale, 'sys.reloadMcpConfirmRequired'))

              return
            }

            if (r.status === 'reloaded') {
              ctx.transcript.sys(
                params.always
                  ? translate(ctx.ui.locale, 'sys.reloadMcpReloaded')
                  : translate(ctx.ui.locale, 'sys.reloadMcpReloadedSimple')
              )

              return
            }

            ctx.transcript.sys(translate(ctx.ui.locale, 'sys.reloadComplete'))
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 're-read ~/.hermes/.env into the running gateway (CLI parity)',
    name: 'reload',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<ReloadEnvResponse>('reload.env', {})
        .then(
          ctx.guarded<ReloadEnvResponse>(r => {
            const n = Number(r.updated ?? 0)
            const noun = n === 1 ? 'var' : 'vars'

            ctx.transcript.sys(translate(ctx.ui.locale, 'sys.reloadEnv', { count: String(n), noun }))
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'manage browser CDP connection [connect|disconnect|status]',
    name: 'browser',
    run: (arg, ctx) => {
      const [rawAction = 'status', ...rest] = arg.trim().split(/\s+/).filter(Boolean)
      const action = rawAction.toLowerCase()

      if (!['connect', 'disconnect', 'status'].includes(action)) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageBrowser'))
      }

      const sid = ctx.sid ?? null
      const url = action === 'connect' ? rest.join(' ').trim() || 'http://127.0.0.1:9222' : undefined

      if (url) {
        ctx.transcript.sys(translate(ctx.ui.locale, 'sys.browserChecking', { url }))
      }

      ctx.gateway
        .rpc<BrowserManageResponse>('browser.manage', { action, session_id: sid, ...(url && { url }) })
        .then(
          ctx.guarded<BrowserManageResponse>(r => {
            // Without a session we can't subscribe to streamed
            // browser.progress events, so flush the bundled list.
            if (!sid) {
              r.messages?.forEach(message => ctx.transcript.sys(message))
            }

            if (action === 'status') {
              return ctx.transcript.sys(
                r.connected
                  ? translate(ctx.ui.locale, 'sys.browserConnectedStatus', {
                      url: r.url || translate(ctx.ui.locale, 'sys.browserUrlUnavailable')
                    })
                  : translate(ctx.ui.locale, 'sys.browserNotConnected')
              )
            }

            if (action === 'disconnect') {
              return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.browserDisconnected'))
            }

            if (r.connected) {
              ctx.transcript.sys(translate(ctx.ui.locale, 'sys.browserConnected'))
              ctx.transcript.sys(
                translate(ctx.ui.locale, 'sys.browserEndpoint', {
                  url: r.url || translate(ctx.ui.locale, 'sys.browserUrlUnavailable')
                })
              )
              ctx.transcript.sys(translate(ctx.ui.locale, 'sys.browserNextCall'))
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'list, diff, or restore checkpoints',
    name: 'rollback',
    run: (arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.rollbackNoSession'))
      }

      const trimmed = arg.trim()
      const [first = '', ...rest] = trimmed.split(/\s+/).filter(Boolean)
      const lower = first.toLowerCase()

      if (!trimmed || lower === 'list' || lower === 'ls') {
        return ctx.gateway
          .rpc<RollbackListResponse>('rollback.list', { session_id: ctx.sid })
          .then(
            ctx.guarded<RollbackListResponse>(r => {
              if (!r.enabled) {
                return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.rollbackNoCheckpoints'))
              }

              const checkpoints = r.checkpoints ?? []

              if (!checkpoints.length) {
                return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.rollbackNone'))
              }

              ctx.transcript.panel(translate(ctx.ui.locale, 'section.rollbackCheckpoints'), [
                {
                  rows: checkpoints.map((c, idx) => [
                    `${idx + 1}. ${c.hash.slice(0, 10)}`,
                    [c.timestamp, c.message].filter(Boolean).join(' · ') ||
                      translate(ctx.ui.locale, 'sys.rollbackNoMetadata')
                  ])
                }
              ])
            })
          )
          .catch(ctx.guardedErr)
      }

      if (lower === 'diff') {
        const hash = rest[0]

        if (!hash) {
          return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageRollbackDiff'))
        }

        return ctx.gateway
          .rpc<RollbackDiffResponse>('rollback.diff', { hash, session_id: ctx.sid })
          .then(
            ctx.guarded<RollbackDiffResponse>(r => {
              const body = (r.rendered || r.diff || '').trim()

              if (!body && !r.stat) {
                return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.rollbackNoChanges'))
              }

              const text = [r.stat || '', body].filter(Boolean).join('\n\n')
              ctx.transcript.page(text, translate(ctx.ui.locale, 'section.rollbackDiff'))
            })
          )
          .catch(ctx.guardedErr)
      }

      const hash = first
      const filePath = rest.join(' ').trim()

      return ctx.gateway
        .rpc<RollbackRestoreResponse>('rollback.restore', {
          ...(filePath ? { file_path: filePath } : {}),
          hash,
          session_id: ctx.sid
        })
        .then(
          ctx.guarded<RollbackRestoreResponse>(r => {
            if (!r.success) {
              return ctx.transcript.sys(
                translate(ctx.ui.locale, 'sys.rollbackFailed', {
                  error: r.error || r.message || translate(ctx.ui.locale, 'common.unknownError')
                })
              )
            }

            const target = filePath || translate(ctx.ui.locale, 'sys.rollbackWorkspace')

            const detail =
              r.reason || r.message || r.restored_to || translate(ctx.ui.locale, 'sys.rollbackRestoredDefault')

            ctx.transcript.sys(translate(ctx.ui.locale, 'sys.rollbackRestored', { target, detail }))

            if ((r.history_removed ?? 0) > 0) {
              ctx.transcript.setHistoryItems(prev => ctx.transcript.trimLastExchange(prev))
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    aliases: ['tasks'],
    help: 'open the spawn-tree dashboard (live audit + kill/pause controls)',
    name: 'agents',
    run: (arg, ctx) => {
      const sub = arg.trim().toLowerCase()

      // Stay compatible with the gateway `/agents [pause|resume|status]` CLI —
      // explicit subcommands skip the overlay and act directly so scripts and
      // multi-step flows can drive it without entering interactive mode.
      if (sub === 'pause' || sub === 'resume' || sub === 'unpause') {
        const paused = sub === 'pause'
        ctx.gateway.gw
          .request<DelegationPauseResponse>('delegation.pause', { paused })
          .then(r => {
            applyDelegationStatus({ paused: r?.paused })
            ctx.transcript.sys(
              r?.paused
                ? translate(ctx.ui.locale, 'sys.delegationPaused')
                : translate(ctx.ui.locale, 'sys.delegationResumed')
            )
          })
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'status') {
        const d = getDelegationState()
        ctx.transcript.sys(
          translate(ctx.ui.locale, 'sys.agentsDelegationStatus', {
            status: translate(ctx.ui.locale, d.paused ? 'common.paused' : 'common.active'),
            maxDepth: String(d.maxSpawnDepth ?? '?'),
            maxConcurrent: String(d.maxConcurrentChildren ?? '?')
          })
        )

        return
      }

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })
    }
  },

  {
    aliases: ['learning', 'memory-graph'],
    help: 'open your learning journey — skills + memories on a timeline',
    name: 'journey',
    run: (_arg, ctx) => {
      void ctx
      patchOverlayState({ journey: true })
    }
  },

  {
    help: 'replay a completed spawn tree · `/replay [N|last|list|load <path>]`',
    name: 'replay',
    run: (arg, ctx) => {
      const history = getSpawnHistory()
      const raw = arg.trim()
      const lower = raw.toLowerCase()

      // ── Disk-backed listing ─────────────────────────────────────
      if (lower === 'list' || lower === 'ls') {
        ctx.gateway
          .rpc<SpawnTreeListResponse>('spawn_tree.list', {
            limit: 30,
            session_id: ctx.sid ?? 'default'
          })
          .then(
            ctx.guarded<SpawnTreeListResponse>(r => {
              const entries = r.entries ?? []

              if (!entries.length) {
                return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.replayNoArchive'))
              }

              const rows: [string, string][] = entries.map(e => {
                const ts = e.finished_at ? new Date(e.finished_at * 1000).toLocaleString() : '?'
                const label = e.label || `${e.count} subagents`

                return [`${ts} · ${e.count}×`, `${label}\n  ${e.path}`]
              })

              ctx.transcript.panel(translate(ctx.ui.locale, 'section.archivedSpawnTrees'), [{ rows }])
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      // ── Disk-backed load by path ─────────────────────────────────
      if (lower.startsWith('load ')) {
        const path = raw.slice(5).trim()

        if (!path) {
          return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageReplayLoad'))
        }

        ctx.gateway
          .rpc<SpawnTreeLoadResponse>('spawn_tree.load', { path })
          .then(
            ctx.guarded<SpawnTreeLoadResponse>(r => {
              if (!r.subagents?.length) {
                return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.replayEmpty'))
              }

              // Push onto the in-memory history so the overlay picks it up
              // by index 1 just like any other snapshot.
              pushDiskSnapshot(r, path)
              patchOverlayState({ agents: true, agentsInitialHistoryIndex: 1 })
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      // ── In-memory nav (same-session) ─────────────────────────────
      if (!history.length) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.replayNoCompleted'))
      }

      let index = 1

      if (raw && lower !== 'last') {
        const parsed = parseInt(raw, 10)

        if (Number.isNaN(parsed) || parsed < 1 || parsed > history.length) {
          return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.replayOutOfRange', { max: String(history.length) }))
        }

        index = parsed
      }

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: index })
    }
  },

  {
    help: 'diff two completed spawn trees · `/replay-diff <baseline> <candidate>` (indexes from /replay list or history N)',
    name: 'replay-diff',
    run: (arg, ctx) => {
      const parts = arg.trim().split(/\s+/).filter(Boolean)

      if (parts.length !== 2) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageReplayDiff'))
      }

      const [a, b] = parts
      const history = getSpawnHistory()

      const resolve = (token: string): null | SpawnSnapshot => {
        const n = parseInt(token!, 10)

        if (Number.isFinite(n) && n >= 1 && n <= history.length) {
          return history[n - 1] ?? null
        }

        return null
      }

      const baseline = resolve(a!)
      const candidate = resolve(b!)

      if (!baseline || !candidate) {
        return ctx.transcript.sys(
          translate(ctx.ui.locale, 'sys.replayDiffCouldNotResolve', { count: String(history.length) })
        )
      }

      setDiffPair({ baseline, candidate })
      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })
    }
  },

  {
    aliases: ['reload_skills'],
    help: 're-scan installed skills in the live TUI gateway',
    name: 'reload-skills',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<SkillsReloadResponse>('skills.reload', {})
        .then(
          ctx.guarded<SkillsReloadResponse>(r => {
            ctx.transcript.page(
              r.output || translate(ctx.ui.locale, 'sys.skillsReloaded'),
              translate(ctx.ui.locale, 'section.reloadSkills')
            )
            ctx.gateway
              .rpc<CommandsCatalogResponse>('commands.catalog', {})
              .then(
                ctx.guarded<CommandsCatalogResponse>(catalog => {
                  if (!catalog?.pairs) {
                    return
                  }

                  ctx.local.setCatalog({
                    canon: (catalog.canon ?? {}) as Record<string, string>,
                    categories: catalog.categories ?? [],
                    pairs: catalog.pairs as [string, string][],
                    skillCount: (catalog.skill_count ?? 0) as number,
                    sub: (catalog.sub ?? {}) as Record<string, string[]>
                  })
                })
              )
              .catch(() => {})
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'browse, inspect, install skills',
    name: 'skills',
    run: (arg, ctx, cmd) => {
      const text = arg.trim()

      if (!text) {
        return patchOverlayState({ skillsHub: true })
      }

      const [sub, ...rest] = text.split(/\s+/)
      const query = rest.join(' ').trim()
      const { rpc } = ctx.gateway
      const { panel, sys } = ctx.transcript

      const runViaSlashWorker = () => {
        ctx.gateway.gw
          .request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: ctx.sid })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            const body = r?.output || '/skills: no output'
            const formatted = r?.warning ? `warning: ${r.warning}\n${body}` : body
            const long = formatted.length > 180 || formatted.split('\n').filter(Boolean).length > 2

            long
              ? ctx.transcript.page(formatted, translate(ctx.ui.locale, 'section.skills'))
              : ctx.transcript.sys(formatted)
          })
          .catch(ctx.guardedErr)
      }

      if (sub === 'list') {
        rpc<SkillsListResponse>('skills.manage', { action: 'list' })
          .then(
            ctx.guarded<SkillsListResponse>(r => {
              const cats = Object.entries(r.skills ?? {}).sort()

              if (!cats.length) {
                return sys(translate(ctx.ui.locale, 'sys.noSkills'))
              }

              panel(
                translate(ctx.ui.locale, 'section.skills'),
                cats.map<PanelSection>(([title, items]) => ({ items, title }))
              )
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'inspect') {
        if (!query) {
          return sys(translate(ctx.ui.locale, 'sys.usageSkillsInspect'))
        }

        rpc<SkillsInspectResponse>('skills.manage', { action: 'inspect', query })
          .then(
            ctx.guarded<SkillsInspectResponse>(r => {
              const info = r.info ?? {}

              if (!info.name) {
                return sys(translate(ctx.ui.locale, 'sys.skillsUnknown', { name: query }))
              }

              const rows: [string, string][] = [
                [translate(ctx.ui.locale, 'field.name'), String(info.name)],
                [translate(ctx.ui.locale, 'field.category'), String(info.category ?? '')],
                [translate(ctx.ui.locale, 'field.path'), String(info.path ?? '')]
              ]

              const sections: PanelSection[] = [{ rows }]

              if (info.description) {
                sections.push({ text: String(info.description) })
              }

              panel(translate(ctx.ui.locale, 'section.skill'), sections)
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'search') {
        if (!query) {
          return sys(translate(ctx.ui.locale, 'sys.usageSkillsSearch'))
        }

        rpc<SkillsSearchResponse>('skills.manage', { action: 'search', query })
          .then(
            ctx.guarded<SkillsSearchResponse>(r => {
              const results = r.results ?? []

              if (!results.length) {
                return sys(translate(ctx.ui.locale, 'sys.skillsNoResults', { query }))
              }

              panel(translate(ctx.ui.locale, 'section.search', { query }), [
                { rows: results.map(s => [s.name, s.description ?? '']) }
              ])
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'install') {
        if (!query) {
          return sys(translate(ctx.ui.locale, 'sys.usageSkillsInstall'))
        }

        sys(translate(ctx.ui.locale, 'sys.skillsInstalling', { name: query }))

        rpc<SkillsInstallResponse>('skills.manage', { action: 'install', query })
          .then(
            ctx.guarded<SkillsInstallResponse>(r =>
              sys(
                r.installed
                  ? translate(ctx.ui.locale, 'sys.skillsInstalled', { name: r.name ?? query })
                  : translate(ctx.ui.locale, 'sys.skillsInstallFailed')
              )
            )
          )
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'browse') {
        const pageNum = query ? parseInt(query, 10) : 1

        if (Number.isNaN(pageNum) || pageNum < 1) {
          return sys(translate(ctx.ui.locale, 'sys.usageSkillsBrowse'))
        }

        sys(translate(ctx.ui.locale, 'sys.skillsFetching'))

        rpc<SkillsBrowseResponse>('skills.manage', { action: 'browse', page: pageNum })
          .then(
            ctx.guarded<SkillsBrowseResponse>(r => {
              const items = r.items ?? []

              if (!items.length) {
                return sys(
                  translate(ctx.ui.locale, 'sys.skillsNoPage', {
                    page: String(pageNum),
                    total: r.total ? ` (total ${r.total})` : ''
                  })
                )
              }

              const rows: [string, string][] = items.map(s => [
                s.trust ? `${s.name} · ${s.trust}` : s.name,
                String(s.description ?? '').slice(0, 160)
              ])

              const footer: string[] = []

              if (r.page && r.total_pages) {
                footer.push(
                  translate(ctx.ui.locale, 'sys.pageOf', { page: String(r.page), total: String(r.total_pages) })
                )
              }

              if (r.total) {
                footer.push(translate(ctx.ui.locale, 'sys.skillsTotal', { total: String(r.total) }))
              }

              if (r.page && r.total_pages && r.page < r.total_pages) {
                footer.push(translate(ctx.ui.locale, 'sys.browseMore', { page: String(r.page + 1) }))
              }

              panel(translate(ctx.ui.locale, 'section.browseSkills'), [
                { rows },
                ...(footer.length ? [{ text: footer.join(' · ') }] : [])
              ])
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      runViaSlashWorker()
    }
  },

  {
    help: 'view & toggle plugins (no arg opens the hub; enable/disable <name> for direct toggle)',
    name: 'plugins',
    run: (arg, ctx, cmd) => {
      // No argument → open the interactive Plugins Hub overlay. Any
      // subcommand (enable/disable/list/install/…) falls through to the
      // text slash worker so it stays at parity with `hermes plugins`.
      if (!arg.trim()) {
        return patchOverlayState({ pluginsHub: true })
      }

      ctx.gateway.gw
        .request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: ctx.sid })
        .then(r => {
          if (ctx.stale()) {
            return
          }

          const body = r?.output || '/plugins: no output'
          const text = r?.warning ? `warning: ${r.warning}\n${body}` : body
          const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2

          long ? ctx.transcript.page(text, 'Plugins') : ctx.transcript.sys(text)
        })
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'enable or disable tools (client-side history reset on change)',
    name: 'tools',
    run: (arg, ctx, cmd) => {
      const [subcommand, ...names] = arg.trim().split(/\s+/).filter(Boolean)

      if (subcommand !== 'disable' && subcommand !== 'enable') {
        ctx.gateway.gw
          .request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: ctx.sid })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            const body = r?.output || '/tools: no output'
            const text = r?.warning ? `warning: ${r.warning}\n${body}` : body
            const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2

            long ? ctx.transcript.page(text, translate(ctx.ui.locale, 'section.tools')) : ctx.transcript.sys(text)
          })
          .catch(ctx.guardedErr)

        return
      }

      if (!names.length) {
        ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageTools', { subcommand }))
        ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageToolsBuiltin', { subcommand }))
        ctx.transcript.sys(translate(ctx.ui.locale, 'sys.usageToolsMcp', { subcommand }))

        return
      }

      ctx.gateway
        .rpc<ToolsConfigureResponse>('tools.configure', { action: subcommand, names, session_id: ctx.sid })
        .then(
          ctx.guarded<ToolsConfigureResponse>(r => {
            if (r.info) {
              ctx.session.setSessionStartedAt(Date.now())
              ctx.session.resetVisibleHistory(r.info)
            }

            if (r.changed?.length) {
              ctx.transcript.sys(
                translate(ctx.ui.locale, 'sys.toolsChanged', {
                  action: translate(ctx.ui.locale, subcommand === 'disable' ? 'common.disabled' : 'common.enabled'),
                  names: r.changed.join(', ')
                })
              )
            }

            if (r.unknown?.length) {
              ctx.transcript.sys(translate(ctx.ui.locale, 'sys.toolsUnknown', { names: r.unknown.join(', ') }))
            }

            if (r.missing_servers?.length) {
              ctx.transcript.sys(
                translate(ctx.ui.locale, 'sys.toolsMissingServers', { names: r.missing_servers.join(', ') })
              )
            }

            if (r.reset) {
              ctx.transcript.sys(translate(ctx.ui.locale, 'sys.toolsSessionReset'))
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
