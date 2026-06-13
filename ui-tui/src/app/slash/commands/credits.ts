import type { CreditsViewResponse } from '../../../gatewayTypes.js'
import { openExternalUrl } from '../../../lib/openExternalUrl.js'
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand } from '../types.js'

export const creditsCommands: SlashCommand[] = [
  {
    help: 'Show Nous credit balance and top up',
    name: 'credits',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<CreditsViewResponse>('credits.view', { session_id: ctx.sid })
        .then(
          ctx.guarded<CreditsViewResponse>(view => {
            if (!view.logged_in) {
              ctx.transcript.sys('💳 Not logged into Nous Portal — run /portal to log in.')

              return
            }

            const lines = ['💳 Nous credits', ...view.balance_lines]

            if (view.identity_line) {
              lines.push('', view.identity_line)
            }

            if (view.topup_url) {
              lines.push('', `Top up: ${view.topup_url}`)
            }

            ctx.transcript.sys(lines.join('\n'))

            const url = view.topup_url

            if (url) {
              patchOverlayState({
                confirm: {
                  cancelLabel: 'Cancel',
                  confirmLabel: 'Open top-up in browser',
                  detail: url,
                  onConfirm: () => {
                    const ok = openExternalUrl(url)
                    ctx.transcript.sys(
                      ok
                        ? 'Complete your top-up in the browser — credits will appear in /credits shortly.'
                        : `Open this URL to top up: ${url}`
                    )
                  },
                  title: 'Add credits?'
                }
              })
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
