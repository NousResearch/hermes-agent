import { attachedFileNotice } from '../../../domain/messages.js'
import type { FileAttachResponse } from '../../../gatewayTypes.js'
import type { SlashCommand } from '../types.js'

export const attachCommands: SlashCommand[] = [
  {
    help: 'attach a file (image, PDF, source, etc.) to the session',
    name: 'attach',
    aliases: ['file'],
    run: (arg, ctx) => {
      if (!arg.trim()) {
        return ctx.transcript.sys('/attach <path>  (use TAB for completion)')
      }
      ctx.gateway.rpc<FileAttachResponse>('file.attach', {
        path: arg,
        session_id: ctx.sid
      }).then(
        ctx.guarded<FileAttachResponse>(r => {
          if (r.remainder) {
            ctx.composer.setInput(r.remainder)
          }
          ctx.transcript.sys(attachedFileNotice(r))
        })
      )
    }
  }
]
