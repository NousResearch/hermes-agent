// FPS counter overlay — renders in the bottom-right corner when
// HERMES_TUI_FPS=1.  Zero-cost when disabled (returns null at the
// top of the component; React skips the whole subtree).
//
// Subscribes to $fpsState via nanostores.  The store is only updated
// when the env flag is on (trackFrame is undefined otherwise), so we
// also gate the subscription on SHOW_FPS to avoid a useless listener.

import { Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { SHOW_FPS } from '../config/env.js'
import { $fpsState } from '../lib/fpsStore.js'

const fpsColor = (fps: number) => {
  if (fps >= 50) {
    return 'green'
  }

  if (fps >= 30) {
    return 'yellow'
  }

  return 'red'
}

export function FpsOverlay() {
  if (!SHOW_FPS) {
    return null
  }

  return <FpsOverlayInner />
}

function FpsOverlayInner() {
  const { fps, lastDurationMs, totalFrames } = useStore($fpsState)

  // Zero-pad to stable width so the corner doesn't jitter as digits
  // come and go.  Format: " 62fps  0.3ms #12345"
  const fpsStr = fps.toFixed(1).padStart(5)
  const durStr = lastDurationMs.toFixed(1).padStart(5)

  return (
    <Text color={fpsColor(fps)}>
      {fpsStr}fps · {durStr}ms · #{totalFrames}
    </Text>
  )
}
