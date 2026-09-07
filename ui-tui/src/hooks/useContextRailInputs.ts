import { execFile } from 'node:child_process'
import { resolve } from 'node:path'

import { useEffect, useState } from 'react'

import { loadRailInputs, type RailInputs } from '../domain/railInputs.js'

export interface RailFlowStatus {
  at: number
  text: string
}

export interface ContextRailData {
  flowStatus: RailFlowStatus | null
  railInputs: RailInputs | null
}

const FLOW_TIMEOUT_MS = 4_000
const REFRESH_INTERVAL_MS = 5_000

export function useContextRailInputs(startDir: string): ContextRailData {
  const [railInputs, setRailInputs] = useState<RailInputs | null>(null)
  const [flowStatus, setFlowStatus] = useState<RailFlowStatus | null>(null)
  const cwd = resolve(startDir || process.cwd())

  useEffect(() => {
    let stopped = false
    setRailInputs(null)
    setFlowStatus(null)

    const refreshRailInputs = () => {
      void loadRailInputs(cwd).then(value => {
        if (!stopped) {
          setRailInputs(value)
        }
      })
    }

    refreshRailInputs()
    const interval = setInterval(refreshRailInputs, REFRESH_INTERVAL_MS)

    return () => {
      stopped = true
      clearInterval(interval)
    }
  }, [cwd])

  useEffect(() => {
    let stopped = false
    const flowRoot = railInputs?.projectRoot
    const flow = railInputs?.flow.trim() ?? ''

    if (!flowRoot || !flow || flow === 'none') {
      return () => {
        stopped = true
      }
    }

    let child: ReturnType<typeof execFile> | null = null

    const refreshFlowStatus = () => {
      child?.kill()
      child = execFile(
        'flowctl',
        ['status'],
        { cwd: flowRoot, maxBuffer: 16 * 1024, timeout: FLOW_TIMEOUT_MS },
        (error, stdout) => {
          const text = error ? 'not connected' : stdout.split(/\r?\n/).slice(0, 2).join(' ').trim() || 'not connected'

          if (!stopped) {
            setFlowStatus({ at: Date.now(), text })
          }
        }
      )
    }

    refreshFlowStatus()
    const interval = setInterval(refreshFlowStatus, REFRESH_INTERVAL_MS)

    return () => {
      stopped = true
      clearInterval(interval)
      child?.kill()
    }
  }, [railInputs])

  return { flowStatus, railInputs }
}
