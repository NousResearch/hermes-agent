import { Box, type ScrollBoxHandle, Text } from '@hermes/ink'
import { type RefObject, useState } from 'react'

import type { Theme } from '../theme.js'

import { scrollbarColors } from './overlayPrimitives.js'

/**
 * Mouse-draggable scrollbar bound to a `ScrollBox` ref. Re-renders off the
 * parent `tick` so accordions / async content can resize the thumb without a
 * scroll event. Shared by every full-screen overlay that scrolls a pane.
 */
export function OverlayScrollbar({
  scrollRef,
  t,
  tick
}: {
  scrollRef: RefObject<null | ScrollBoxHandle>
  t: Theme
  tick: number
}) {
  void tick

  const [hover, setHover] = useState(false)
  const [grab, setGrab] = useState<null | number>(null)

  const s = scrollRef.current
  const vp = Math.max(0, s?.getViewportHeight() ?? 0)

  if (!vp) {
    return <Box width={1} />
  }

  const total = Math.max(vp, s?.getScrollHeight() ?? vp)
  const scrollable = total > vp
  const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp
  const travel = Math.max(1, vp - thumb)
  const pos = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0))
  const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0
  const below = Math.max(0, vp - thumbTop - thumb)

  const vBar = (n: number) => (n > 0 ? `${'│\n'.repeat(n - 1)}│` : '')
  const thumbBody = `${'┃\n'.repeat(Math.max(0, thumb - 1))}┃`
  const { thumb: thumbColor, track: trackColor } = scrollbarColors(t, hover, grab !== null)

  const jump = (row: number, offset: number) => {
    if (!s || !scrollable) {
      return
    }

    s.scrollTo(Math.round((Math.max(0, Math.min(travel, row - offset)) / travel) * Math.max(0, total - vp)))
  }

  return (
    <Box
      flexDirection="column"
      onMouseDown={(e: { localRow?: number }) => {
        const row = Math.max(0, Math.min(vp - 1, e.localRow ?? 0))
        const off = row >= thumbTop && row < thumbTop + thumb ? row - thumbTop : Math.floor(thumb / 2)
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grab ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => setGrab(null)}
      width={1}
    >
      {!scrollable ? (
        <Text color={trackColor}>{vBar(vp)}</Text>
      ) : (
        <>
          {thumbTop > 0 ? <Text color={trackColor}>{vBar(thumbTop)}</Text> : null}

          <Text color={thumbColor}>{thumbBody}</Text>

          {below > 0 ? <Text color={trackColor}>{vBar(below)}</Text> : null}
        </>
      )}
    </Box>
  )
}
