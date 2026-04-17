import type { ToolsConfigureResponse } from '../../../gatewayTypes.js'
import type { SlashCommand } from '../types.js'

export const opsCommands: SlashCommand[] = [
  {
    help: 'enable or disable tools (client-side history reset on change)',
    name: 'tools',
    run: (arg, ctx) => {
      const [subcommand, ...names] = arg.trim().split(/\s+/).filter(Boolean)

      if (subcommand !== 'disable' && subcommand !== 'enable') {
        return
      }

      if (!names.length) {
        ctx.transcript.sys(`usage: /tools ${subcommand} <name> [name ...]`)
        ctx.transcript.sys(`built-in toolset: /tools ${subcommand} web`)
        ctx.transcript.sys(`MCP tool: /tools ${subcommand} github:create_issue`)

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
              ctx.transcript.sys(`${subcommand === 'disable' ? 'disabled' : 'enabled'}: ${r.changed.join(', ')}`)
            }

            if (r.unknown?.length) {
              ctx.transcript.sys(`unknown toolsets: ${r.unknown.join(', ')}`)
            }

            if (r.missing_servers?.length) {
              ctx.transcript.sys(`missing MCP servers: ${r.missing_servers.join(', ')}`)
            }

            if (r.reset) {
              ctx.transcript.sys('session reset. new tool configuration is active.')
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
