import type { RunExternalProcess } from '@hermes/ink'

import type { SetupStatusResponse } from '../gatewayTypes.js'
import { t } from '../i18n/runtime.js'
import type { LaunchResult } from '../lib/externalCli.js'

import type { SlashHandlerContext } from './interfaces.js'
import { patchUiState } from './uiStore.js'

export interface RunExternalSetupOptions {
  args: string[]
  ctx: Pick<SlashHandlerContext, 'gateway' | 'session' | 'transcript'>
  done: string
  launcher: (args: string[]) => Promise<LaunchResult>
  suspend: (run: RunExternalProcess) => Promise<void>
}

export async function runExternalSetup({ args, ctx, done, launcher, suspend }: RunExternalSetupOptions) {
  const { gateway, session, transcript } = ctx

  transcript.sys(t('session.handoff.launching', args.join(' ')))
  patchUiState({ status: t('session.status.setupRunning') })

  let result: LaunchResult = { code: null }

  await suspend(async () => {
    result = await launcher(args)
  })

  if (result.error) {
    transcript.sys(t('session.handoff.launchError', result.error))
    patchUiState({ status: t('session.status.setupRequired') })

    return
  }

  if (result.code !== 0) {
    transcript.sys(t('session.handoff.exitedWithCode', args[0], result.code))
    patchUiState({ status: t('session.status.setupRequired') })

    return
  }

  const setup = await gateway.rpc<SetupStatusResponse>('setup.status', {})

  if (setup?.provider_configured === false) {
    transcript.sys(t('session.handoff.stillNoProvider'))
    patchUiState({ status: t('session.status.setupRequired') })

    return
  }

  transcript.sys(done)
  session.newSession()
}
