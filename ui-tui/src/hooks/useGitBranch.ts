import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

import { useEffect, useState } from 'react'

const TTL_MS = 15_000
const TIMEOUT_MS = 500

const pexec = promisify(execFile)
const cache = new Map<string, { at: number; branch: null | string }>()
const inflight = new Map<string, Promise<null | string>>()

const resolveBranch = async (cwd: string): Promise<null | string> => {
  try {
    const { stdout } = await pexec('git', ['-C', cwd, 'rev-parse', '--abbrev-ref', 'HEAD'], { timeout: TIMEOUT_MS })
    const b = stdout.trim()

    return !b || b === 'HEAD' ? null : b
  } catch {
    return null
  }
}

const fetchBranch = (cwd: string): Promise<null | string> => {
  const pending = inflight.get(cwd)

  if (pending) {
    return pending
  }

  const p = resolveBranch(cwd).finally(() => inflight.delete(cwd))
  inflight.set(cwd, p)

  return p
}

export function useGitBranch(cwd: string): null | string {
  const [branch, setBranch] = useState<null | string>(() => cache.get(cwd)?.branch ?? null)

  useEffect(() => {
    let cancelled = false

    const tick = async () => {
      const hit = cache.get(cwd)

      if (hit && Date.now() - hit.at < TTL_MS) {
        if (!cancelled) {
          setBranch(hit.branch)
        }

        return
      }

      const b = await fetchBranch(cwd)
      cache.set(cwd, { at: Date.now(), branch: b })

      if (!cancelled) {
        setBranch(b)
      }
    }

    void tick()
    const id = setInterval(() => void tick(), TTL_MS)

    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [cwd])

  return branch
}
