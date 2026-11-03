// FPS counter overlay (HERMES_TUI_FPS=1). Zero-cost when disabled.

import { Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { SHOW_FPS } from '../config/env.js'
import { $fpsState } from '../lib/fpsStore.js'
import type { Theme } from '../theme.js'

const fpsColor = (fps: number, t: Theme) =>
  fps >= 50 ? t.color.statusGood : fps >= 30 ? t.color.statusWarn : t.color.error

export function FpsOverlay({ t }: { t: Theme }) {
  if (!SHOW_FPS) {
    return null
  }

  return <FpsOverlayInner t={t} />
}

function FpsOverlayInner({ t }: { t: Theme }) {
  const { fps, lastDurationMs, totalFrames } = useStore($fpsState)

  // Zero-pad widths so digit churn doesn't jitter the corner.
  return (
    <Text color={fpsColor(fps, t)}>
      {fps.toFixed(1).padStart(5)}fps · {lastDurationMs.toFixed(1).padStart(5)}ms · #{totalFrames}
    </Text>
  )
}
