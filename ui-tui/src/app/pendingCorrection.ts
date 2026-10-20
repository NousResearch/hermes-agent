export type CorrectionSurface = 'redirect' | 'steer'

export type AcceptedCorrectionView = 'none' | 'steer-row' | 'user-row'

/**
 * Where an accepted busy correction becomes visible.
 *
 * A steer — including an interrupt that the agent stored as a steer because a
 * tool was running — is the queue strip's first row. A model redirect is a
 * user row in the live turn. Standalone interrupt already paints that row on
 * the owned-turn path, so only the canonical gateway needs it here.
 */
export function acceptedCorrectionView(
  mode: 'interrupt' | 'queue' | 'steer',
  canonical: boolean,
  pending?: CorrectionSurface | null
): AcceptedCorrectionView {
  if (pending === 'redirect') {
    return canonical ? 'user-row' : 'none'
  }

  if (pending === 'steer' || mode === 'steer' || (canonical && mode === 'interrupt')) {
    return 'steer-row'
  }

  return 'none'
}

/** Reconnect snapshot: take the authority's steer only when this client has none yet. */
export function steerToHydrate(
  local: null | string,
  snapshot: { steer?: null | string } | null | undefined
): null | string {
  if (local) {
    return null
  }

  return snapshot?.steer || null
}
