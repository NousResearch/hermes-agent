import { useStdout } from '@hermes/ink'
import { useCallback, useEffect, useRef, useState } from 'react'

import type { PetGrid } from '../components/petSprite.js'

import { useGateway } from './gatewayContext.js'
import { $turnState } from './turnStore.js'
import { $uiState } from './uiStore.js'

export type PetState = 'idle' | 'wave' | 'run' | 'failed' | 'review' | 'jump'

interface PetActivity {
  busy: boolean
  toolRunning: boolean
  reasoning: boolean
}

/**
 * Resolve the animation state — mirrors `agent.pet.state.derive_pet_state`
 * (and the desktop's `derivePetState`) so all surfaces agree.
 */
export function derivePetState({ busy, toolRunning, reasoning }: PetActivity): PetState {
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
 * live. The frame cache is keyed by `slug:state` so a switch re-pulls cleanly.
 */
export function usePet(): PetRender {
  const { rpc } = useGateway()
  const { write } = useStdout()
  const [enabled, setEnabled] = useState(false)
  const [grid, setGrid] = useState<PetGrid | null>(null)
  const [kitty, setKitty] = useState<KittyView | null>(null)

  const cache = useRef<Map<string, CacheEntry>>(new Map())
  const slugRef = useRef('')
  const imageIdRef = useRef(0)
  const stateRef = useRef<PetState>('idle')
  const frameRef = useRef(0)

  const [petState, setPetState] = useState<PetState>('idle')

  // Recompute the desired state on every turn/ui change.
  useEffect(() => {
    const recompute = () => {
      const turn = $turnState.get()
      const ui = $uiState.get()

      const next = derivePetState({
        busy: ui.busy,
        toolRunning: turn.tools.length > 0,
        reasoning: turn.reasoningActive
      })

      if (next !== stateRef.current) {
        stateRef.current = next
        frameRef.current = 0
        setPetState(next)
      }
    }

    recompute()
    const unsubTurn = $turnState.listen(recompute)
    const unsubUi = $uiState.listen(recompute)

    return () => {
      unsubTurn()
      unsubUi()
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

  // Fetch + cache one (slug, state). `pet.cells` resolves the active pet from
  // config, so its `slug`/`enabled` are the source of truth.
  const sync = useCallback(
    async (state: PetState) => {
      try {
        const res = (await rpc('pet.cells', { graphics: IS_TTY, state })) as PetCellsResult | null

        if (!res) {
          return
        }

        if (!res.enabled) {
          releaseKitty()
          slugRef.current = ''
          cache.current.clear()
          setGrid(null)
          setKitty(null)
          setEnabled(false)

          return
        }

        const slug = res.slug ?? ''

        if (slug !== slugRef.current) {
          releaseKitty()
          slugRef.current = slug
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
      } catch {
        // cosmetic — ignore RPC failures
      }
    },
    [rpc, releaseKitty]
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
