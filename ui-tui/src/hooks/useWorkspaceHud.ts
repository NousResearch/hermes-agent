import { useEffect, useMemo, useState } from 'react'

import {
  EMPTY_GIT_WORKSPACE,
  type GitWorkspaceSnapshot,
  parseWorkspaceInfo,
  type WorkspaceHudSnapshot
} from '../domain/workspaceHud.js'
import type { GatewayClient } from '../gatewayClient.js'
import { asRpcResult } from '../lib/rpc.js'

export const WORKSPACE_HUD_POLL_MS = 12_000
const WORKSPACE_REQUEST_TIMEOUT_MS = 2_500

interface CacheEntry {
  at: number
  value: GitWorkspaceSnapshot | null
}

const cache = new WeakMap<GatewayClient, Map<string, CacheEntry>>()
const pending = new WeakMap<GatewayClient, Map<string, Promise<GitWorkspaceSnapshot | null>>>()

const cacheFor = (gw: GatewayClient) => {
  const existing = cache.get(gw)

  if (existing) {
    return existing
  }

  const created = new Map<string, CacheEntry>()
  cache.set(gw, created)

  return created
}

const pendingFor = (gw: GatewayClient) => {
  const existing = pending.get(gw)

  if (existing) {
    return existing
  }

  const created = new Map<string, Promise<GitWorkspaceSnapshot | null>>()
  pending.set(gw, created)

  return created
}

const requestWorkspaceInfo = async (gw: GatewayClient, cwd: string): Promise<GitWorkspaceSnapshot | null> => {
  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    const result = await Promise.race([
      gw.request<unknown>('workspace.info', { cwd }),
      new Promise<null>(resolve => {
        timer = setTimeout(() => resolve(null), WORKSPACE_REQUEST_TIMEOUT_MS)
        timer.unref?.()
      })
    ])

    const payload = asRpcResult(result)

    return payload ? parseWorkspaceInfo(payload) : null
  } catch {
    return null
  } finally {
    if (timer) {
      clearTimeout(timer)
    }
  }
}

const fetchWorkspaceInfo = (gw: GatewayClient, cwd: string): Promise<GitWorkspaceSnapshot | null> => {
  const key = cwd.trim()
  const now = Date.now()
  const gatewayCache = cacheFor(gw)
  const gatewayPending = pendingFor(gw)
  const hit = gatewayCache.get(key)

  if (hit && now - hit.at < WORKSPACE_HUD_POLL_MS) {
    return Promise.resolve(hit.value)
  }

  const inFlight = gatewayPending.get(key)

  if (inFlight) {
    return inFlight
  }

  const request = requestWorkspaceInfo(gw, key)
    .then(value => {
      gatewayCache.set(key, { at: Date.now(), value })

      return value
    })
    .catch(() => {
      gatewayCache.set(key, { at: Date.now(), value: null })

      return null
    })
    .finally(() => {
      gatewayPending.delete(key)
    })

  gatewayPending.set(key, request)

  return request
}

const initialSnapshot = (fallbackBranch: null | string | undefined): GitWorkspaceSnapshot => ({
  ...EMPTY_GIT_WORKSPACE,
  branch: fallbackBranch?.trim() || null
})

/** Backend-aware Git/GitHub metadata for the read-only Ink footer HUD. */
export function useWorkspaceHud(
  gw: GatewayClient,
  cwd: string,
  projectName?: null | string,
  fallbackBranch?: null | string
): WorkspaceHudSnapshot {
  const [git, setGit] = useState<GitWorkspaceSnapshot>(() => initialSnapshot(fallbackBranch))
  const normalizedCwd = cwd.trim()
  const normalizedFallbackBranch = fallbackBranch?.trim() || null

  useEffect(() => {
    let stopped = false
    setGit(initialSnapshot(normalizedFallbackBranch))

    if (!normalizedCwd) {
      return () => {
        stopped = true
      }
    }

    const refresh = () => {
      void fetchWorkspaceInfo(gw, normalizedCwd).then(value => {
        if (!stopped && value) {
          setGit(value)
        }
      })
    }

    refresh()
    const timer = setInterval(refresh, WORKSPACE_HUD_POLL_MS)

    return () => {
      stopped = true
      clearInterval(timer)
    }
  }, [gw, normalizedCwd, normalizedFallbackBranch])

  return useMemo(
    () => ({
      ...git,
      projectName: projectName?.trim() || null
    }),
    [git, projectName]
  )
}
