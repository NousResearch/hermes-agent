import { Box, Text } from '@hermes/ink'
import { memo } from 'react'

// A cell is [tr,tg,tb,ta, br,bg,bb,ba] — the top + bottom pixel of one
// half-block, as produced by the `pet.cells` gateway RPC.
export type PetCell = number[]
export type PetGrid = PetCell[][]

const UPPER_HALF = '▀'
const LOWER_HALF = '▄'

const hex = (r: number, g: number, b: number) =>
  `#${[r, g, b].map(v => Math.max(0, Math.min(255, v | 0)).toString(16).padStart(2, '0')).join('')}`

/**
 * Renders one petdex frame as truecolor half-blocks using native Ink color
 * props (no raw ANSI, so width measurement stays correct). The engine
 * (`agent/pet/render.py`) does the decode + downscale; this is a thin painter.
 */
export const PetSprite = memo(function PetSprite({ grid }: { grid: PetGrid }) {
  if (!grid.length) {
    return null
  }

  return (
    <Box flexDirection="column">
      {grid.map((row, y) => (
        <Box key={y}>
          {row.map((cell, x) => {
            const [tr, tg, tb, ta, br, bg, bb, ba] = cell
            const top = (ta ?? 0) >= 32
            const bot = (ba ?? 0) >= 32

            if (!top && !bot) {
              return <Text key={x}> </Text>
            }

            // Both halves opaque → fg=top over bg=bottom. One half opaque →
            // draw it fg-only so the other stays the terminal bg (no black
            // boxes bleeding around transparent sprite edges).
            if (top && bot) {
              return (
                <Text backgroundColor={hex(br, bg, bb)} color={hex(tr, tg, tb)} key={x}>
                  {UPPER_HALF}
                </Text>
              )
            }

            return top ? (
              <Text color={hex(tr, tg, tb)} key={x}>
                {UPPER_HALF}
              </Text>
            ) : (
              <Text color={hex(br, bg, bb)} key={x}>
                {LOWER_HALF}
              </Text>
            )
          })}
        </Box>
      ))}
    </Box>
  )
})

/**
 * Renders a kitty Unicode-placeholder grid: each line is a row of U+10EEEE
 * cells whose foreground color encodes the image id. The actual pixels are
 * drawn by the terminal (the frame image is transmitted out-of-band by
 * `usePet`); this only emits the placeholder text Ink can measure as width-1
 * cells. Truecolor-only — the color must reach the terminal verbatim for the
 * id to decode, which Ghostty/kitty support.
 */
export const PetKitty = memo(function PetKitty({
  color,
  placeholder
}: {
  color: string
  placeholder: string[]
}) {
  if (!placeholder.length) {
    return null
  }

  return (
    <Box flexDirection="column">
      {placeholder.map((row, y) => (
        <Text color={color} key={y}>
          {row}
        </Text>
      ))}
    </Box>
  )
})
