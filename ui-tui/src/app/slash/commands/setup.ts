import { withInkSuspended } from '@hermes/ink'

import { translate } from '../../../i18n/index.js'
import { launchHermesCommand } from '../../../lib/externalCli.js'
import { runExternalSetup } from '../../setupHandoff.js'
import type { SlashCommand } from '../types.js'

export const setupCommands: SlashCommand[] = [
  {
    name: 'setup',
    run: (arg, ctx) =>
      void runExternalSetup({
        args: ['setup', ...arg.split(/\s+/).filter(Boolean)],
        ctx,
        done: translate(ctx.ui.locale, 'setup.complete'),
        launcher: launchHermesCommand,
        locale: ctx.ui.locale,
        suspend: withInkSuspended
      })
  }
]
