import { Box, useStdout } from '@hermes/ink'
import { memo, useEffect, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import { TERMUX_TUI_MODE } from '../config/env.js'
import { INLINE_IMAGE_MAX_COLS, INLINE_IMAGE_MAX_ROWS } from '../config/limits.js'
import type { ImageRenderResponse } from '../gatewayTypes.js'
import { transcriptBodyWidth, transcriptGutterWidth } from '../lib/inputMetrics.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

import { HalfBlockImage, KittyImage } from './cellImage.js'

// Only a TUI that owns a real terminal can take the kitty transmit; piped or
// under the dashboard PTY the gateway answers with half-blocks.
const IS_TTY = Boolean(process.stdout?.isTTY)

// Renders outlive virtual-scroll remounts. kitty keeps a transmitted image
// under its id, so each (image, size) goes over the tty once per process.
const renders = new Map<string, Promise<ImageRenderResponse | null>>()
const transmitted = new Set<string>()

const requestRender = (gw: ReturnType<typeof useGateway>['gw'], path: string, cols: number) => {
  const key = `${path}\0${cols}`
  let pending = renders.get(key)

  if (!pending) {
    pending = gw
      .request('image.render', { cols, graphics: IS_TTY, path, rows: INLINE_IMAGE_MAX_ROWS })
      .then(res => asRpcResult<ImageRenderResponse>(res))
      .catch(() => null)
      .then(res => {
        if (!res) {
          renders.delete(key) // a dropped request retries on the next mount
        }

        return res
      })
    renders.set(key, pending)
  }

  return pending
}

/**
 * A local image drawn inline in the transcript: kitty Unicode placeholders
 * where the terminal paints them (kitty, Ghostty), truecolor half-blocks
 * everywhere else. Both are ordinary cells, so Ink measures and scrolls them
 * like text. The engine (`agent/terminal_image.py`) picks the mode from
 * `display.inline_images` and does the decode and layout.
 */
export const InlineImage = memo(function InlineImage({ cols, path }: { cols: number; path: string }) {
  const { gw } = useGateway()
  const { write } = useStdout()
  const [image, setImage] = useState<ImageRenderResponse | null>(null)
  const budget = Math.max(1, Math.min(INLINE_IMAGE_MAX_COLS, cols))

  useEffect(() => {
    let live = true

    void requestRender(gw, path, budget).then(res => {
      if (!live || !res) {
        return
      }

      const sent = `${res.color}:${res.cols}x${res.rows}`

      if (res.mode === 'kitty' && res.transmit && !transmitted.has(sent)) {
        try {
          write(res.transmit)
          transmitted.add(sent)
        } catch {
          // a dead tty paints nothing either way
        }
      }

      setImage(res)
    })

    return () => {
      live = false
    }
  }, [budget, gw, path, write])

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

/** A transcript row's images (`msg.images`), stacked at the body indent; nothing when it has none. */
export const MessageImages = ({ cols, msg, spaced = false, t }: MessageImagesProps) =>
  msg.images?.length ? (
    <Box flexDirection="column" marginLeft={transcriptGutterWidth(msg.role, t.brand.prompt)} marginTop={spaced ? 1 : 0}>
      {msg.images.map(path => (
        <InlineImage
          cols={transcriptBodyWidth(cols, msg.role, t.brand.prompt, TERMUX_TUI_MODE)}
          key={path}
          path={path}
        />
      ))}
    </Box>
  ) : null
