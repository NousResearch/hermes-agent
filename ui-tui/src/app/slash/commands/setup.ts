import { withInkSuspended } from '@hermes/ink'

import { launchHermesCommand } from '../../../lib/externalCli.js'
import { runExternalSetup } from '../../setupHandoff.js'
import type { SlashCommand } from '../types.js'

export const setupCommands: SlashCommand[] = [
  {
    help: 'configure LLM provider + model (launches `hermes model`)',
    name: 'provider',
    run: (_arg, ctx) =>
      void runExternalSetup({
        args: ['model'],
        ctx,
        done: 'provider updated — starting session…',
        launcher: launchHermesCommand,
        suspend: withInkSuspended
      })
  },
  {
    help: 'run full setup wizard (launches `hermes setup`)',
    name: 'setup',
    run: (arg, ctx) =>
      void runExternalSetup({
        args: ['setup', ...arg.split(/\s+/).filter(Boolean)],
        ctx,
        done: 'setup complete — starting session…',
        launcher: launchHermesCommand,
        suspend: withInkSuspended
      })
  }
]
