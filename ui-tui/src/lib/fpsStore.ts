// Tiny FPS tracker fed by ink's onFrame callback.
//
// Keeps a ring buffer of the last N frame timestamps and derives fps
// from the rolling window.  Updates a nanostore so a corner-overlay
// component can subscribe without pulling it through props.
//
// FPS here means "Ink render rate" — each entry is an ink frame, which
// includes both React commits and drain-only frames (Ink re-rendering
// with an updated scrollTop without a React commit).  That's the right
// notion for user-perceived motion: it's how often the screen buffer
// actually changes, not how often React reconciles.
//
// Zero-cost when HERMES_TUI_FPS is unset: trackFrame is undefined so
// the onFrame callback short-circuits at the optional chain.

import { atom } from 'nanostores'

import { SHOW_FPS } from '../config/env.js'

const WINDOW_SIZE = 30 // last 30 frames

export type FpsState = {
  /** Frames per second averaged over the last WINDOW_SIZE frames. */
  fps: number
  /** Total frames counted since start (wraps at JS-safe int so you can
   *  diff pairs in a debug overlay without worrying about precision). */
  totalFrames: number
  /** Last frame's durationMs (ink render phase total). */
  lastDurationMs: number
}

export const $fpsState = atom<FpsState>({
  fps: 0,
  lastDurationMs: 0,
  totalFrames: 0
})

const timestamps: number[] = []
let totalFrames = 0

export const trackFrame = SHOW_FPS
  ? (durationMs: number) => {
      const now = performance.now()

      timestamps.push(now)

      if (timestamps.length > WINDOW_SIZE) {
        timestamps.shift()
      }

      totalFrames++

      // FPS = frames-in-window / seconds-in-window. Needs at least 2
      // timestamps to compute a gap.
      if (timestamps.length >= 2) {
        const elapsed = (timestamps[timestamps.length - 1]! - timestamps[0]!) / 1000

        if (elapsed > 0) {
          const fps = (timestamps.length - 1) / elapsed

          $fpsState.set({
            fps: Math.round(fps * 10) / 10,
            lastDurationMs: Math.round(durationMs * 100) / 100,
            totalFrames
          })
        }
      }
    }
  : undefined
