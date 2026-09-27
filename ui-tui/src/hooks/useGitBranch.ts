import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

import { useEffect, useState } from 'react'

const TTL_MS = 5_000
const TIMEOUT_MS = 750

const pexec = promisify(execFile)

export interface GitStatus {
  ahead: number
  behind: number
  branch: null | string
  conflicted: number
  modified: number
  staged: number
  untracked: number
}

export const EMPTY_GIT_STATUS: GitStatus = {
  ahead: 0,
  behind: 0,
  branch: null,
  conflicted: 0,
  modified: 0,
  staged: 0,
  untracked: 0
}

const cache = new Map<string, { at: number; status: GitStatus }>()
const inflight = new Map<string, Promise<GitStatus>>()

// One `--porcelain=v2 --branch` read supplies branch plus every dirty/divergence
// counter. `--no-optional-locks` keeps a status-bar repaint from contending with
// a concurrent git process (same reason the Claude status line uses it).
export const parseGitStatus = (text: string): GitStatus => {
  const result = { ...EMPTY_GIT_STATUS }

  for (const line of text.split('\n')) {
    if (line.startsWith('# branch.head ')) {
      const branch = line.slice('# branch.head '.length).trim()

      result.branch = branch === '(detached)' ? 'detached' : branch || null
    } else if (line.startsWith('# branch.ab ')) {
      const match = line.match(/\+(\d+)\s+-(\d+)/)

      if (match) {
        result.ahead = Number(match[1])
        result.behind = Number(match[2])
      }
    } else if (line.startsWith('1 ') || line.startsWith('2 ')) {
      const xy = line.split(/\s+/)[1] ?? '..'

      if ((xy[0] ?? '.') !== '.') {
        result.staged += 1
      }

      if ((xy[1] ?? '.') !== '.') {
        result.modified += 1
      }
    } else if (line.startsWith('u ')) {
      result.conflicted += 1
    } else if (line.startsWith('? ')) {
      result.untracked += 1
    }
  }

  return result
}

const resolveStatus = async (cwd: string): Promise<GitStatus> => {
  try {
    const { stdout } = await pexec(
      'git',
      ['-C', cwd, '--no-optional-locks', 'status', '--porcelain=v2', '--branch'],
      { timeout: TIMEOUT_MS }
    )

    return parseGitStatus(stdout)
  } catch {
    return { ...EMPTY_GIT_STATUS }
  }
}

const fetchStatus = (cwd: string): Promise<GitStatus> => {
  const pending = inflight.get(cwd)

  if (pending) {
    return pending
  }

  const request = resolveStatus(cwd).finally(() => inflight.delete(cwd))
  inflight.set(cwd, request)

  return request
}

export function useGitStatus(cwd: string): GitStatus {
  const [status, setStatus] = useState<GitStatus>(() => cache.get(cwd)?.status ?? { ...EMPTY_GIT_STATUS })

  useEffect(() => {
    let cancelled = false

    const tick = async () => {
      const hit = cache.get(cwd)

      if (hit && Date.now() - hit.at < TTL_MS) {
        if (!cancelled) {
          setStatus(hit.status)
        }

        return
      }

      const next = await fetchStatus(cwd)
      cache.set(cwd, { at: Date.now(), status: next })

      if (!cancelled) {
        setStatus(next)
      }
    }

    void tick()
    const id = setInterval(() => void tick(), TTL_MS)

    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [cwd])

  return status
}

export function useGitBranch(cwd: string): null | string {
  return useGitStatus(cwd).branch
}
