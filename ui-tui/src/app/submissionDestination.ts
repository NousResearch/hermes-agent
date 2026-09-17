import { realpathSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'

import { getUiState } from './uiStore.js'

export interface SubmissionDestination {
  readonly sid: string | null
  readonly storedSid?: string | null
  readonly profile: string
  readonly profileHome: string
}

// Resolve existing ancestors too: a new profile may not have been created yet.
function canonicalHome(path: string): string {
  try {
    return realpathSync(path)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }

    const parent = dirname(path)

    return parent === path ? path : join(canonicalHome(parent), basename(path))
  }
}

export function captureDestination(): SubmissionDestination {
  const { sid, info } = getUiState()

  return Object.freeze({
    sid,
    storedSid: info?.stored_session_id || null,
    profile: info?.profile_name || 'default',
    profileHome: canonicalHome(resolve(process.env.HERMES_HOME ?? join(homedir(), '.hermes')))
  })
}

export function isCurrentDestination(destination: SubmissionDestination): boolean {
  const current = captureDestination()

  return (
    current.sid === destination.sid &&
    current.profile === destination.profile &&
    current.profileHome === destination.profileHome
  )
}
