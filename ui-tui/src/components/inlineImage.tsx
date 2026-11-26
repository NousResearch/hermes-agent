import { Box, useStdout } from '@hermes/ink'
import type { ImageRenderResult } from '@hermes/shared/gateway-events'
import { useStore } from '@nanostores/react'
import { memo, useEffect, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import { $uiInlineImages } from '../app/uiStore.js'
import { TERMUX_TUI_MODE } from '../config/env.js'
import { INLINE_IMAGE_MAX_COLS, INLINE_IMAGE_MAX_ROWS } from '../config/limits.js'
import { transcriptBodyWidth, transcriptGutterWidth } from '../lib/inputMetrics.js'
import { asRpcResult } from '../lib/rpc.js'
import { detectTerminalGraphics } from '../lib/terminalGraphics.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

import { HalfBlockImage, KittyImage } from './cellImage.js'

// Only a TUI that owns a real terminal can take the kitty transmit; piped or
// under the dashboard PTY the gateway answers with half-blocks.
const IS_TTY = Boolean(process.stdout?.isTTY)
const TERMINAL = detectTerminalGraphics()

// Renders outlive virtual-scroll remounts. The key carries a revision (the
// tool call id, or the user turn) so a second picture at an old path is a new request; `off`
// is never kept, so a file that appears later or a config change still shows.
// A small LRU: kitty entries drop their transmit once written, but half-block
// grids stay resident.
const RENDER_CACHE_MAX = 48

interface RenderEntry {
  pending: Promise<ImageRenderResult | null>
  /** Set once resolved, so a remount paints on its first frame. */
  result?: ImageRenderResult
}

const renders = new Map<string, RenderEntry>()

// kitty holds one virtual placement per image id, and every size of a file
// shares that id (the placeholder color), so transmitting size B replaces
// size A. The size the terminal holds now, per id.
const placed = new Map<string, string>()

const sizeOf = (res: ImageRenderResult) => `${res.cols}x${res.rows}`

const renderKey = (path: string, revision: string | undefined, cols: number) => `${path}\0${revision ?? ''}\0${cols}`

/** A resolved render that paints correctly right now: for kitty, the terminal must hold its size. */
const usableRender = (key: string): ImageRenderResult | null => {
  const result = renders.get(key)?.result

  if (!result || (result.mode === 'kitty' && placed.get(result.color ?? '') !== sizeOf(result))) {
    return null
  }

  return result
}

const remember = (key: string, entry: RenderEntry) => {
  renders.delete(key)
  renders.set(key, entry)

  for (const oldest of renders.keys()) {
    if (renders.size <= RENDER_CACHE_MAX) {
      break
    }

    renders.delete(oldest)
  }
}

const requestRender = (
  gw: ReturnType<typeof useGateway>['gw'],
  write: (data: string) => boolean,
  path: string,
  revision: string | undefined,
  cols: number
) => {
  const key = renderKey(path, revision, cols)
  const cached = renders.get(key)

  // In flight, or still paintable. A kitty entry another size displaced has
  // no transmit left, so it asks again.
  if (cached && (!cached.result || usableRender(key))) {
    remember(key, cached)

    return cached.pending
  }

  const entry: RenderEntry = {
    pending: gw
      .request('image.render', { cols, graphics: IS_TTY, path, rows: INLINE_IMAGE_MAX_ROWS, terminal: TERMINAL })
      .then(res => asRpcResult<ImageRenderResult>(res))
      .catch(() => null)
      .then(res => {
        if (!res || res.mode === 'off') {
          // A dropped request or `off` retries on the next mount.
          if (renders.get(key) === entry) {
            renders.delete(key)
          }

          return res
        }

        // Single-flight: every mount of this key shares this one write.
        if (res.mode === 'kitty' && res.transmit) {
          try {
            write(res.transmit)
            placed.set(res.color ?? '', sizeOf(res))
          } catch {
            // a dead tty paints nothing either way
          }

          res = { ...res, transmit: undefined }
        }

        entry.result = res

        return res
      })
  }

  remember(key, entry)

  return entry.pending
}

interface InlineImageProps {
  cols: number
  path: string
  /** Tells two pictures at one path apart (tool call id or user turn); see `Msg.imageRevision`. */
  revision?: string
}

/**
 * A local image drawn inline in the transcript: kitty Unicode placeholders
 * where the terminal paints them (kitty, Ghostty), truecolor half-blocks
 * everywhere else. Both are ordinary cells, so Ink measures and scrolls them
 * like text. The engine (`agent/terminal_image.py`) picks the mode from
 * `display.inline_images` and does the decode and layout.
 */
export const InlineImage = memo(function InlineImage({ cols, path, revision }: InlineImageProps) {
  const { gw } = useGateway()
  const { write } = useStdout()
  const budget = Math.max(1, Math.min(INLINE_IMAGE_MAX_COLS, cols))
  const key = renderKey(path, revision, budget)
  // Seeded from a resolved render so a virtual-scroll remount has its real height on the first frame.
  const [image, setImage] = useState<ImageRenderResult | null>(() => usableRender(key))

  useEffect(() => {
    let live = true

    void requestRender(gw, write, path, revision, budget).then(res => {
      if (live && res) {
        setImage(res)
      }
    })

    return () => {
      live = false
    }
  }, [budget, gw, path, revision, write])

  if (image?.mode === 'kitty' && image.color && image.placeholder?.length) {
    return <KittyImage color={image.color} placeholder={image.placeholder} />
  }

  if (image?.mode === 'unicode' && image.cells?.length) {
    return <HalfBlockImage grid={image.cells} />
  }

  return null
})

interface MessageImagesProps {
  cols: number
  msg: Msg
  /** A blank line between the row's text and its images. */
  spaced?: boolean
  t: Theme
}

/**
 * A transcript row's images (`msg.images`), stacked at the body indent; nothing
 * when it has none or `display.inline_images` is off (the height estimate
 * reserves nothing then either: lib/virtualHeights.ts).
 */
export const MessageImages = ({ cols, msg, spaced = false, t }: MessageImagesProps) => {
  const enabled = useStore($uiInlineImages)

  return enabled && msg.images?.length ? (
    <Box flexDirection="column" marginLeft={transcriptGutterWidth(msg.role, t.brand.prompt)} marginTop={spaced ? 1 : 0}>
      {msg.images.map(path => (
        <InlineImage
          cols={transcriptBodyWidth(cols, msg.role, t.brand.prompt, TERMUX_TUI_MODE)}
          key={path}
          path={path}
          revision={msg.imageRevision}
        />
      ))}
    </Box>
  ) : null
}
