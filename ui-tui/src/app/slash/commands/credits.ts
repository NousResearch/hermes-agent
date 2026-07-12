import type { CreditsViewResponse } from '../../../gatewayTypes.js'
import { translate } from '../../../i18n/index.js'
import { openExternalUrl } from '../../../lib/openExternalUrl.js'
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand } from '../types.js'

export const creditsCommands: SlashCommand[] = [
  {
    name: 'credits',
    run: (_arg, ctx) => {
      const t = (key: Parameters<typeof translate>[1], vars?: Record<string, string | number>) =>
        translate(ctx.ui.locale, key, vars)

      ctx.gateway
        .rpc<CreditsViewResponse>('credits.view', { session_id: ctx.sid })
        .then(
          ctx.guarded<CreditsViewResponse>(view => {
            if (!view.logged_in) {
              ctx.transcript.sys(t('credits.notLoggedIn'))

              return
            }

            const lines = [t('credits.title'), ...view.balance_lines]

            if (view.identity_line) {
              lines.push('', view.identity_line)
            }

            if (view.topup_url) {
              lines.push('', t('credits.topUp', { url: view.topup_url }))
            }

            ctx.transcript.sys(lines.join('\n'))

            const url = view.topup_url

            if (url) {
              patchOverlayState({
                confirm: {
                  cancelLabel: t('common.cancel'),
                  confirmLabel: t('credits.openTopUp'),
                  detail: url,
                  onConfirm: () => {
                    const ok = openExternalUrl(url)
                    ctx.transcript.sys(ok ? t('credits.completeInBrowser') : t('credits.openUrl', { url }))
                  },
                  title: t('credits.addTitle')
                }
              })
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
