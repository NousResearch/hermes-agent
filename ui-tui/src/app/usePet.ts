import { useStdout } from '@hermes/ink'
import { useCallback, useEffect, useRef, useState } from 'react'

import type { PetGrid } from '../components/petSprite.js'
import { createPetSingleFlight, requestPetUpdate } from '../lib/petPolling.js'

import { useGateway } from './gatewayContext.js'
import { $overlayState, getOverlayState } from './overlayStore.js'
import { $petFlash } from './petFlashStore.js'
import { $turnState } from './turnStore.js'
import { $uiState } from './uiStore.js'

export type PetState = 'idle' | 'wave' | 'run' | 'failed' | 'review' | 'jump' | 'waiting'

interface PetActivity {
  busy: boolean
  toolRunning: boolean
  reasoning: boolean
  awaitingInput: boolean
}

/**
 * Resolve the animation state — mirrors `agent.pet.state.derive_pet_state`
 * (and the desktop's `derivePetState`) so all surfaces agree. `awaitingInput`
 * (a clarify/approval blocking on the user) outranks the in-flight signals
 * because the turn is paused on you, not working.
 */
export function derivePetState({ busy, toolRunning, reasoning, awaitingInput }: PetActivity): PetState {
  if (awaitingInput) {
    return 'waiting'
  }

  if (toolRunning) {
    return 'run'
  }

  if (reasoning) {
    return 'review'
  }

  if (busy) {
    return 'run'
  }

  return 'idle'
}

// The overlays that mean "the agent is blocked on the user" (vs. user-toggled
// pickers like model/sessions, which aren't the agent waiting).
function isAwaitingInput(): boolean {
  const o = getOverlayState()

  return Boolean(o.clarify || o.approval || o.sudo || o.secret || o.confirm)
}

// A kitty Unicode-placeholder frame set: a static placeholder grid (painted by
// Ink in the image-id color) plus per-frame transmit escapes written straight
// to the terminal out-of-band.
interface KittyView {
  color: string
  placeholder: string[]
}

interface PetCellsResult {
  color?: string
  enabled?: boolean
  frameMs?: number
  // unicode mode: cell grids; kitty mode: transmit-escape strings.
  frames?: PetGrid[] | string[]
  graphics?: string
  imageId?: number
  placeholder?: string[]
  scale?: number
  slug?: string
  state?: string
}

type CacheEntry =
  | { kind: 'cells'; frameMs: number; frames: PetGrid[] }
  | { kind: 'kitty'; frameMs: number; frames: string[]; placeholder: string[]; color: string }

const FRAME_MS = 160
const POLL_MS = 2500

// Only the standalone TUI owns a real terminal it can splat image escapes into;
// when piped (or running under the dashboard PTY the gateway resolves to
// half-blocks anyway) we never ask for graphics.
const IS_TTY = Boolean(process.stdout?.isTTY)

export interface PetRender {
  enabled: boolean
  grid: PetGrid | null
  kitty: KittyView | null
}

/**
 * Drives the TUI pet. Fetches each (slug, state)'s frames via the `pet.cells`
 * RPC (cached) and animates the frame index. Two render paths:
 *
 * - **kitty** (Ghostty/kitty): the engine returns a static placeholder grid +
 *   per-frame transmit escapes. We paint the placeholder with Ink and write the
 *   current frame's escape to the terminal out-of-band, so the image animates
 *   underneath without Ink ever repainting.
 * - **cells** (everywhere else): truecolor half-block grids painted by Ink.
 *
 * A steady poll keeps it reactive to config changes made elsewhere (`/pet`, the
 * picker, `hermes pets select`) so adopting/switching/disabling takes effect
 * live. Disabled/cached pets use the cheap inline `pet.info.meta` probe; only
 * uncached enabled states request `pet.cells` from the long-handler pool.
 */
export function usePet(): PetRender {
  const { gw } = useGateway()
  const { write } = useStdout()
  const [enabled, setEnabled] = useState(false)
  const [grid, setGrid] = useState<PetGrid | null>(null)
  const [kitty, setKitty] = useState<KittyView | null>(null)

  const cache = useRef<Map<string, CacheEntry>>(new Map())
  const slugRef = useRef('')
  const scaleRef = useRef(0)
  const revisionRef = useRef('')
  const imageIdRef = useRef(0)
  const stateRef = useRef<PetState>('idle')
  const frameRef = useRef(0)
  const runSingleFlight = useRef(createPetSingleFlight()).current

  const [petState, setPetState] = useState<PetState>('idle')

  // Recompute the desired state on every turn/ui/flash change. A transient
  // flash (wave/jump/failed) wins until it expires; a timer re-runs at expiry.
  useEffect(() => {
    let expiry: ReturnType<typeof setTimeout> | undefined

    const apply = (next: PetState) => {
      if (next !== stateRef.current) {
        stateRef.current = next
        frameRef.current = 0
        setPetState(next)
      }
    }

    const recompute = () => {
      clearTimeout(expiry)

      const flash = $petFlash.get()
      const now = Date.now()

      if (flash && now < flash.until) {
        apply(flash.state)
        expiry = setTimeout(recompute, flash.until - now)

        return
      }

      const turn = $turnState.get()
      const ui = $uiState.get()

      apply(
        derivePetState({
          awaitingInput: isAwaitingInput(),
          busy: ui.busy,
          reasoning: turn.reasoningActive,
          toolRunning: turn.tools.length > 0
        })
      )
    }

    recompute()
    const unsubTurn = $turnState.listen(recompute)
    const unsubUi = $uiState.listen(recompute)
    const unsubFlash = $petFlash.listen(recompute)
    const unsubOverlay = $overlayState.listen(recompute)

    return () => {
      clearTimeout(expiry)
      unsubTurn()
      unsubUi()
      unsubFlash()
      unsubOverlay()
    }
  }, [])

  // Free the terminal-side image when the pet goes away or the hook unmounts.
  const releaseKitty = useCallback(() => {
    if (imageIdRef.current) {
      try {
        write(`\x1b_Ga=d,d=i,i=${imageIdRef.current},q=2\x1b\\`)
      } catch {
        // best-effort cleanup
      }

      imageIdRef.current = 0
    }
  }, [write])

  const disablePet = useCallback(() => {
    releaseKitty()
    slugRef.current = ''
    scaleRef.current = 0
    revisionRef.current = ''
    cache.current.clear()
    setGrid(null)
    setKitty(null)
    setEnabled(false)
  }, [releaseKitty])

  // Probe the active selection cheaply, then fetch + cache one uncached state.
  const sync = useCallback(
    (state: PetState) =>
      runSingleFlight(async () => {
        const update = await requestPetUpdate<PetCellsResult>(gw, state, IS_TTY, meta => {
          const slug = meta.slug ?? ''
          const scale = meta.scale ?? 0
          const revision = meta.spritesheetRevision ?? ''

          const selectionChanged =
            slug !== slugRef.current || scale !== scaleRef.current || revision !== revisionRef.current

          if (selectionChanged) {
            releaseKitty()
            slugRef.current = slug
            scaleRef.current = scale
            revisionRef.current = revision
            cache.current.clear()
            frameRef.current = 0
          }

          return !cache.current.has(`${slug}:${state}`)
        })

        if (!update) {
          return
        }

        if (!update.meta.enabled) {
          disablePet()

          return
        }

        const res = update.cells

        if (!res) {
          setEnabled(true)

          return
        }

        if (!res.enabled) {
          disablePet()

          return
        }

        const slug = res.slug ?? update.meta.slug ?? ''
        const scale = res.scale ?? update.meta.scale ?? 0

        // Config may change between the metadata and frame calls. Keep the
        // frame response authoritative and force a fresh metadata revision on
        // the next poll when the response moved to another selection.
        if (slug !== slugRef.current || scale !== scaleRef.current) {
          releaseKitty()
          slugRef.current = slug
          scaleRef.current = scale
          revisionRef.current = slug === update.meta.slug ? revisionRef.current : ''
          cache.current.clear()
          frameRef.current = 0
        }

        if (res.graphics === 'kitty' && res.frames?.length && res.placeholder?.length) {
          imageIdRef.current = res.imageId ?? 0
          cache.current.set(`${slug}:${state}`, {
            color: res.color ?? '#000001',
            frameMs: res.frameMs ?? FRAME_MS,
            frames: res.frames as string[],
            kind: 'kitty',
            placeholder: res.placeholder
          })
        } else if (res.frames?.length) {
          cache.current.set(`${slug}:${state}`, {
            frameMs: res.frameMs ?? FRAME_MS,
            frames: res.frames as PetGrid[],
            kind: 'cells'
          })
        }

        setEnabled(true)
      }),
    [disablePet, gw, releaseKitty, runSingleFlight]
  )

  // Pull frames whenever the state changes (if not already cached for the
  // active pet), plus a steady poll that catches adopt/switch/disable.
  useEffect(() => {
    if (!cache.current.has(`${slugRef.current}:${petState}`)) {
      void sync(petState)
    }

    const timer = setInterval(() => void sync(stateRef.current), POLL_MS)

    return () => clearInterval(timer)
  }, [petState, sync])

  useEffect(() => releaseKitty, [releaseKitty])

  // Animation timer.
  useEffect(() => {
    if (!enabled) {
      return
    }

    const tick = () => {
      const entry = cache.current.get(`${slugRef.current}:${stateRef.current}`)

      if (!entry?.frames.length) {
        return // keep the last frame painted while the new state loads
      }

      const idx = frameRef.current % entry.frames.length
      frameRef.current = idx + 1

      if (entry.kind === 'kitty') {
        // Transmit this frame's image under the shared id; the static
        // placeholder cells (set below) render it. No Ink repaint needed.
        try {
          write(entry.frames[idx] ?? '')
        } catch {
          // ignore transmit failures
        }

        setGrid(null)
        setKitty(prev =>
          prev && prev.color === entry.color && prev.placeholder === entry.placeholder
            ? prev
            : { color: entry.color, placeholder: entry.placeholder }
        )

        return
      }

      setKitty(null)
      setGrid(entry.frames[idx] ?? null)
    }

    tick()
    const interval = setInterval(tick, FRAME_MS)

    return () => clearInterval(interval)
  }, [enabled, petState, write])

  return { enabled, grid, kitty }
}
