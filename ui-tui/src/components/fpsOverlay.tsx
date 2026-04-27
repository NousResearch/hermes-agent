// FPS counter overlay (HERMES_TUI_FPS=1). Zero-cost when disabled.

import { Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { SHOW_FPS } from '../config/env.js'
import { $fpsState } from '../lib/fpsStore.js'

const fpsColor = (fps: number) => (fps >= 50 ? 'green' : fps >= 30 ? 'yellow' : 'red')

export function FpsOverlay() {
  if (!SHOW_FPS) {
    return null
  }

  return <FpsOverlayInner />
}

function FpsOverlayInner() {
  const { fps, lastDurationMs, totalFrames } = useStore($fpsState)

  // Zero-pad widths so digit churn doesn't jitter the corner.
  return (
    <Text color={fpsColor(fps)}>
      {fps.toFixed(1).padStart(5)}fps · {lastDurationMs.toFixed(1).padStart(5)}ms · #{totalFrames}
    </Text>
  )
}
