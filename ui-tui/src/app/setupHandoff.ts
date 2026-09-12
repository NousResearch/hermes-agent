import type { RunExternalProcess } from '@hermes/ink'

import { FREE_TIER_LIMIT_KEY } from '../content/setup.js'
import { sessionScopedModelArg } from '../domain/slash.js'
import type { ConfigSetResponse, RuntimeCheckResponse, SetupStatusResponse } from '../gatewayTypes.js'
import type { LaunchResult } from '../lib/externalCli.js'

import type { SlashHandlerContext } from './interfaces.js'
import { freeTierBlockMessage, setFreeTierBlock } from './freeTierGate.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

export interface RunExternalSetupOptions {
  args: string[]
  ctx: Pick<SlashHandlerContext, 'gateway' | 'session' | 'transcript'>
  done: string
  launcher: (args: string[]) => Promise<LaunchResult>
  suspend: (run: RunExternalProcess) => Promise<void>
}

export async function runExternalSetup({ args, ctx, done, launcher, suspend }: RunExternalSetupOptions) {
  const { gateway, session, transcript } = ctx
  const current = getUiState()
  const continuationSession = freeTierBlockMessage() ? current.sid : null

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

  if (continuationSession) {
    if (getUiState().sid !== continuationSession) {
      return
    }
    await gateway.rpc('reload.env', {})
    const runtime = await gateway.rpc<RuntimeCheckResponse>('setup.runtime_check', {})
    if (getUiState().sid !== continuationSession) {
      return
    }
    if (!runtime?.ok || runtime.free_tier !== false || !runtime.model || !runtime.provider) {
      transcript.sys(runtime?.error || 'Choose a local model or configure another provider to continue.')
      patchUiState({ status: 'choose a provider to continue' })
      return
    }
    const configured = await gateway.rpc<ConfigSetResponse>('config.set', {
      key: 'model', session_id: continuationSession,
      value: sessionScopedModelArg(`${runtime.model} --provider ${runtime.provider}`)
    })
    if (getUiState().sid !== continuationSession || !configured?.value || configured.credential_warning || configured.confirm_required) {
      transcript.sys(configured?.credential_warning || configured?.confirm_message || 'Provider setup is not complete yet.')
      return
    }
    setFreeTierBlock(null)
    turnController.clearNotice(FREE_TIER_LIMIT_KEY)
    patchUiState({ status: 'ready', ...(configured.info ? { info: configured.info } : {}) })
    transcript.sys('Provider ready — continue in this conversation.')
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
