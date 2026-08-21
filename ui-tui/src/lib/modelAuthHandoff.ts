import { withInkSuspended } from '@hermes/ink'

import { launchHermesCommand, type LaunchResult } from './externalCli.js'

export interface ModelAuthHandoffOptions {
  launcher?: (args: string[]) => Promise<LaunchResult>
  suspend?: typeof withInkSuspended
}

export async function runModelAuthHandoff({
  launcher = launchHermesCommand,
  suspend = withInkSuspended
}: ModelAuthHandoffOptions = {}): Promise<LaunchResult> {
  let result: LaunchResult = { code: null }

  await suspend(async () => {
    result = await launcher(['model'])
  })

  return result
}
