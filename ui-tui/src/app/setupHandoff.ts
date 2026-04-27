import type { RunExternalProcess } from '@hermes/ink'

import type { SetupStatusResponse } from '../gatewayTypes.js'
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

  transcript.sys(`launching \`hermes ${args.join(' ')}\`…`)
  patchUiState({ status: 'setup running…' })

  let result: LaunchResult = { code: null }

  await suspend(async () => {
    result = await launcher(args)
  })

  if (result.error) {
    transcript.sys(`error launching hermes: ${result.error}`)
    patchUiState({ status: 'setup required' })

    return
  }

  if (result.code !== 0) {
    transcript.sys(`hermes ${args[0]} exited with code ${result.code}`)
    patchUiState({ status: 'setup required' })

    return
  }

  const setup = await gateway.rpc<SetupStatusResponse>('setup.status', {})

  if (setup?.provider_configured === false) {
    transcript.sys('still no provider configured')
    patchUiState({ status: 'setup required' })

    return
  }

  transcript.sys(done)
  session.newSession()
}
