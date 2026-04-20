import { spawn } from 'child_process'
type ExecFileOptions = {
  input?: string
  timeout?: number
  useCwd?: boolean
  env?: NodeJS.ProcessEnv
}

export function execFileNoThrow(
  file: string,
  args: string[],
  options: ExecFileOptions = {}
): Promise<{
  stdout: string
  stderr: string
  code: number
  error?: string
}> {
  return new Promise(resolve => {
    const child = spawn(file, args, {
      cwd: options.useCwd ? process.cwd() : undefined,
      env: options.env,
      stdio: 'pipe'
    })

    let stdout = ''
    let stderr = ''
    let timedOut = false

    const timer = options.timeout
      ? setTimeout(() => {
          timedOut = true
          child.kill('SIGTERM')
        }, options.timeout)
      : null

    child.stdout?.on('data', chunk => {
      stdout += String(chunk)
    })
    child.stderr?.on('data', chunk => {
      stderr += String(chunk)
    })
    child.on('error', error => {
      if (timer) {
        clearTimeout(timer)
      }

      resolve({ stdout, stderr, code: 1, error: String(error) })
    })
    child.on('close', code => {
      if (timer) {
        clearTimeout(timer)
      }

      resolve({ stdout, stderr, code: timedOut ? 124 : (code ?? 0) })
    })

    if (options.input) {
      child.stdin?.write(options.input)
    }

    child.stdin?.end()
  })
}
