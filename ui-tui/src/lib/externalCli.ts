import { spawn } from 'node:child_process'

export interface LaunchResult {
  code: null | number
  error?: string
}

const resolveRabbitBin = () => process.env.RABBIT_BIN?.trim() || 'rabbit'

export const launchRabbitCommand = (args: string[]): Promise<LaunchResult> =>
  new Promise(resolve => {
    const child = spawn(resolveRabbitBin(), args, { stdio: 'inherit' })

    child.on('error', err => resolve({ code: null, error: err.message }))
    child.on('exit', code => resolve({ code }))
  })
