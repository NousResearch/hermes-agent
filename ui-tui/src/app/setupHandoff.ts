import type { RunExternalProcess } from '@hermes/ink'

import type { SetupStatusResponse } from '../gatewayTypes.js'
import { type Locale, translate } from '../i18n/index.js'
import type { LaunchResult } from '../lib/externalCli.js'

import type { SlashHandlerContext } from './interfaces.js'
import { patchUiState } from './uiStore.js'

export interface RunExternalSetupOptions {
  args: string[]
  ctx: Pick<SlashHandlerContext, 'gateway' | 'session' | 'transcript'>
  done: string
  locale: Locale
  launcher: (args: string[]) => Promise<LaunchResult>
  suspend: (run: RunExternalProcess) => Promise<void>
}

export async function runExternalSetup({ args, ctx, done, launcher, locale, suspend }: RunExternalSetupOptions) {
  const { gateway, session, transcript } = ctx

  transcript.sys(translate(locale, 'setup.launching', { args: args.join(' ') }))
  patchUiState({ status: 'setup running…' })

  let result: LaunchResult = { code: null }

  await suspend(async () => {
    result = await launcher(args)
  })

  if (result.error) {
    transcript.sys(translate(locale, 'setup.launchError', { error: result.error }))
    patchUiState({ status: 'setup required' })

    return
  }

  if (result.code !== 0) {
    transcript.sys(translate(locale, 'setup.exitedWithCode', { command: args[0] ?? '', code: String(result.code) }))
    patchUiState({ status: 'setup required' })

    return
  }

  const setup = await gateway.rpc<SetupStatusResponse>('setup.status', {})

  if (setup?.provider_configured === false) {
    transcript.sys(translate(locale, 'setup.stillNoProvider'))
    patchUiState({ status: 'setup required' })

    return
  }

  transcript.sys(done)
  session.newSession()
}
