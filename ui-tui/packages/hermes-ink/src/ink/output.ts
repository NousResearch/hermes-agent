import { type AnsiCode, type StyledChar, styledCharsFromTokens, tokenize } from '@alcalzone/ansi-tokenize'

import { logForDebugging } from '../utils/debug.js'
import { getGraphemeSegmenter } from '../utils/intl.js'
import sliceAnsi from '../utils/sliceAnsi.js'

import { reorderBidi } from './bidi.js'
import { type Rectangle, unionRect } from './layout/geometry.js'
import {
  blitRegion,
  CellWidth,
  extractHyperlinkFromStyles,
  filterOutHyperlinkStyles,
  markNoSelectRegion,
  OSC8_PREFIX,
  resetScreen,
  type Screen,
  setCellAt,
  shiftRows,
  type StylePool
} from './screen.js'
import { stringWidth } from './stringWidth.js'
import { widestLine } from './widest-line.js'

/**
 * A grapheme cluster with precomputed terminal width, styleId, and hyperlink.
 * Built once per unique line (cached via charCache), so the per-char hot loop
 * is just property reads + setCellAt — no stringWidth, no style interning,
 * no hyperlink extraction per frame.
 *
 * styleId is safe to cache: StylePool is session-lived (never reset).
 * hyperlink is stored as a string (not interned ID) since hyperlinkPool
 * resets every 5 min; setCellAt interns it per-frame (cheap Map.get).
 */
type ClusteredChar = {
  value: string
  width: number
  styleId: number
  hyperlink: string | undefined
}

/**
 * Collects write/blit/clear/clip operations from the render tree, then
 * applies them to a Screen buffer in `get()`. The Screen is what gets
 * diffed against the previous frame to produce terminal updates.
 */

type Options = {
  width: number
  height: number
  stylePool: StylePool
  /**
   * Screen to render into. Will be reset before use.
   * For double-buffering, pass a reusable screen. Otherwise create a new one.
   */
  screen: Screen
}

export type Operation =
  | WriteOperation
  | ClipOperation
  | UnclipOperation
  | BlitOperation
  | ClearOperation
  | NoSelectOperation
  | ShiftOperation

type WriteOperation = {
  type: 'write'
  x: number
  y: number
  text: string
  /**
   * Per-line soft-wrap flags, parallel to text.split('\n'). softWrap[i]=true
   * means line i is a continuation of line i-1 (the `\n` before it was
   * inserted by word-wrap, not in the source). Index 0 is always false.
   * Undefined means the producer didn't track wrapping (e.g. fills,
   * raw-ansi) — the screen's per-row bitmap is left untouched.
   */
  softWrap?: boolean[]
}

type ClipOperation = {
  type: 'clip'
  clip: Clip
}

export type Clip = {
  x1: number | undefined
  x2: number | undefined
  y1: number | undefined
  y2: number | undefined
}

/**
 * Intersect two clips. `undefined` on an axis means unbounded; the other
 * clip's bound wins. If both are bounded, take the tighter constraint
 * (max of mins, min of maxes). If the resulting region is empty
 * (x1 >= x2 or y1 >= y2), writes clipped by it will be dropped.
 */
function intersectClip(parent: Clip | undefined, child: Clip): Clip {
  if (!parent) {
    return child
  }

  return {
    x1: maxDefined(parent.x1, child.x1),
    x2: minDefined(parent.x2, child.x2),
    y1: maxDefined(parent.y1, child.y1),
    y2: minDefined(parent.y2, child.y2)
  }
}

function maxDefined(a: number | undefined, b: number | undefined): number | undefined {
  if (a === undefined) {
    return b
  }

  if (b === undefined) {
    return a
  }

  return Math.max(a, b)
}

function minDefined(a: number | undefined, b: number | undefined): number | undefined {
  if (a === undefined) {
    return b
  }

  if (b === undefined) {
    return a
  }

  return Math.min(a, b)
}

type UnclipOperation = {
  type: 'unclip'
}

type BlitOperation = {
  type: 'blit'
  src: Screen
  x: number
  y: number
  width: number
  height: number
}

type ShiftOperation = {
  type: 'shift'
  top: number
  bottom: number
  n: number
}

type ClearOperation = {
  type: 'clear'
  region: Rectangle
  /**
   * Set when the clear is for an absolute-positioned node's old bounds.
   * Absolute nodes overlay normal-flow siblings, so their stale paint is
   * what an earlier sibling's clean-subtree blit wrongly restores from
   * prevScreen. Normal-flow siblings' clears don't have this problem —
   * their old position can't have been painted on top of a sibling.
   */
  fromAbsolute?: boolean
}

type NoSelectOperation = {
  type: 'noSelect'
  region: Rectangle
}

export default class Output {
  width: number
  height: number
  private readonly stylePool: StylePool
  private screen: Screen

  private readonly operations: Operation[] = []

  private charCache: Map<string, ClusteredChar[]> = new Map()

  constructor(options: Options) {
    const { width, height, stylePool, screen } = options

    this.width = width
    this.height = height
    this.stylePool = stylePool
    this.screen = screen

    resetScreen(screen, width, height)
  }

  /**
   * Reuse this Output for a new frame. Zeroes the screen buffer, clears
   * the operation list (backing storage is retained), and caps charCache
   * growth. Preserving charCache across frames is the main win — most
   * lines don't change between renders, so tokenize + grapheme clustering
   * becomes a cache hit.
   */
  reset(width: number, height: number, screen: Screen): void {
    this.width = width
    this.height = height
    this.screen = screen
    this.operations.length = 0
    resetScreen(screen, width, height)

    if (this.charCache.size > 16384) {
      this.charCache.clear()
    }
  }

  /**
   * Copy cells from a source screen region (blit = block image transfer).
   */
  blit(src: Screen, x: number, y: number, width: number, height: number): void {
    this.operations.push({ type: 'blit', src, x, y, width, height })
  }

  /**
   * Shift full-width rows within [top, bottom] by n. n > 0 = up. Mirrors
   * what DECSTBM + SU/SD does to the terminal. Paired with blit() to reuse
   * prevScreen content during pure scroll, avoiding full child re-render.
   */
  shift(top: number, bottom: number, n: number): void {
    this.operations.push({ type: 'shift', top, bottom, n })
  }

  /**
   * Clear a region by writing empty cells. Used when a node shrinks to
   * ensure stale content from the previous frame is removed.
   */
  clear(region: Rectangle, fromAbsolute?: boolean): void {
    this.operations.push({ type: 'clear', region, fromAbsolute })
  }

  /**
   * Mark a region as non-selectable (excluded from fullscreen text
   * selection copy + highlight). Used by <NoSelect> to fence off
   * gutters (line numbers, diff sigils). Applied AFTER blit/write so
   * the mark wins regardless of what's blitted into the region.
   */
  noSelect(region: Rectangle): void {
    this.operations.push({ type: 'noSelect', region })
  }

  write(x: number, y: number, text: string, softWrap?: boolean[]): void {
    if (!text) {
      return
    }

    this.operations.push({
      type: 'write',
      x,
      y,
      text,
      softWrap
    })
  }

  clip(clip: Clip) {
    this.operations.push({
      type: 'clip',
      clip
    })
  }

  unclip() {
    this.operations.push({
      type: 'unclip'
    })
  }

  get(): Screen {
    const screen = this.screen
    const screenWidth = this.width
    const screenHeight = this.height

    // Track blit vs write cell counts for debugging
    let blitCells = 0
    let writeCells = 0

    // Pass 1: expand damage to cover clear regions. The buffer is freshly
    // zeroed by resetScreen, so this pass only marks damage so diff()
    // checks these regions against the previous frame.
    //
    // Also collect clears from absolute-positioned nodes. An absolute
    // node overlays normal-flow siblings; when it shrinks, its clear is
    // pushed AFTER those siblings' clean-subtree blits (DOM order). The
    // blit copies the absolute node's own stale paint from prevScreen,
    // and since clear is damage-only, the ghost survives diff. Normal-
    // flow clears don't need this — a normal-flow node's old position
    // can't have been painted on top of a sibling's current position.
    const absoluteClears: Rectangle[] = []

    for (const operation of this.operations) {
      if (operation.type !== 'clear') {
        continue
      }

      const { x, y, width, height } = operation.region
      const startX = Math.max(0, x)
      const startY = Math.max(0, y)
      const maxX = Math.min(x + width, screenWidth)
      const maxY = Math.min(y + height, screenHeight)

      if (startX >= maxX || startY >= maxY) {
        continue
      }

      const rect = {
        x: startX,
        y: startY,
        width: maxX - startX,
        height: maxY - startY
      }

      screen.damage = screen.damage ? unionRect(screen.damage, rect) : rect

      if (operation.fromAbsolute) {
        absoluteClears.push(rect)
      }
    }

    const clips: Clip[] = []

    for (const operation of this.operations) {
      switch (operation.type) {
        case 'clear':
          // handled in pass 1
          continue

        case 'clip':
          // Intersect with the parent clip (if any) so nested
          // overflow:hidden boxes can't write outside their ancestor's
          // clip region. Without this, a message with overflow:hidden at
          // the bottom of a scrollbox pushes its OWN clip (based on its
          // layout bounds, already translated by -scrollTop) which can
          // extend below the scrollbox viewport — writes escape into
          // the sibling bottom section's rows.
          clips.push(intersectClip(clips.at(-1), operation.clip))

          continue

        case 'unclip':
          clips.pop()

          continue
        case 'blit': {
          // Bulk-copy cells from source screen region using TypedArray.set().
          // Tracking damage ensures diff() checks blitted cells for stale content
          // when a parent blits an area that previously contained child content.
          const { src, x: regionX, y: regionY, width: regionWidth, height: regionHeight } = operation

          // Intersect with active clip — a child's clean-blit passes its full
          // cached rect, but the parent ScrollBox may have shrunk (pill mount).
          // Without this, the blit writes past the ScrollBox's new bottom edge
          // into the pill's row.
          const clip = clips.at(-1)
          const startX = Math.max(regionX, clip?.x1 ?? 0)
          const startY = Math.max(regionY, clip?.y1 ?? 0)

          const maxY = Math.min(regionY + regionHeight, screenHeight, src.height, clip?.y2 ?? Infinity)

          const maxX = Math.min(regionX + regionWidth, screenWidth, src.width, clip?.x2 ?? Infinity)

          if (startX >= maxX || startY >= maxY) {
            continue
          }

          // Exclude cells covered by an absolute-positioned node's clear.
          // Absolute nodes overlay normal-flow siblings, so prevScreen in
          // that region holds stale overlay paint. If we blit those cells
          // back, removed/moved overlays ghost as a duplicate.
          if (absoluteClears.length === 0) {
            blitRegion(screen, src, startX, startY, maxX, maxY)
            blitCells += (maxY - startY) * (maxX - startX)

            continue
          }

          for (let row = startY; row < maxY; row++) {
            let spans: [number, number][] = [[startX, maxX]]

            for (const r of absoluteClears) {
              if (row < r.y || row >= r.y + r.height || !spans.length) {
                break
              }

              const cs = Math.max(startX, r.x)
              const ce = Math.min(maxX, r.x + r.width)

              if (cs >= ce) {
                continue
              }

              const next: [number, number][] = []

              for (const [sx, ex] of spans) {
                if (ce <= sx || cs >= ex) {
                  next.push([sx, ex])

                  continue
                }

                if (sx < cs) {
                  next.push([sx, cs])
                }

                if (ce < ex) {
                  next.push([ce, ex])
                }
              }

              spans = next
            }

            for (const [sx, ex] of spans) {
              blitRegion(screen, src, sx, row, ex, row + 1)
              blitCells += ex - sx
            }
          }

          continue
        }

        case 'shift': {
          shiftRows(screen, operation.top, operation.bottom, operation.n)

          continue
        }

        case 'write': {
          const { text, softWrap } = operation
          let { x, y } = operation
          let lines = text.split('\n')
          let swFrom = 0
          let prevContentEnd = 0

          const clip = clips.at(-1)

          if (clip) {
            const clipHorizontally = typeof clip?.x1 === 'number' && typeof clip?.x2 === 'number'

            const clipVertically = typeof clip?.y1 === 'number' && typeof clip?.y2 === 'number'

            // If text is positioned outside of clipping area altogether,
            // skip to the next operation to avoid unnecessary calculations
            if (clipHorizontally) {
              const width = widestLine(text)

              if (x + width <= clip.x1! || x >= clip.x2!) {
                continue
              }
            }

            if (clipVertically) {
              const height = lines.length

              if (y + height <= clip.y1! || y >= clip.y2!) {
                continue
              }
            }

            if (clipHorizontally) {
              lines = lines.map(line => {
                const from = x < clip.x1! ? clip.x1! - x : 0
                const width = stringWidth(line)
                const to = x + width > clip.x2! ? clip.x2! - x : width
                let sliced = sliceAnsi(line, from, to)

                // Wide chars (CJK, emoji) occupy 2 cells. When `to` lands
                // on the first cell of a wide char, sliceAnsi includes the
                // entire glyph and the result overflows clip.x2 by one cell,
                // writing a SpacerTail into the adjacent sibling. Re-slice
                // one cell earlier; wide chars are exactly 2 cells, so a
                // single retry always fits.
                if (stringWidth(sliced) > to - from) {
                  sliced = sliceAnsi(line, from, to - 1)
                }

                return sliced
              })

              if (x < clip.x1!) {
                x = clip.x1!
              }
            }

            if (clipVertically) {
              const from = y < clip.y1! ? clip.y1! - y : 0
              const height = lines.length
              const to = y + height > clip.y2! ? clip.y2! - y : height

              // If the first visible line is a soft-wrap continuation, we
              // need the clipped previous line's content end so
              // screen.softWrap[lineY] correctly records the join point
              // even though that line's cells were never written.
              if (softWrap && from > 0 && softWrap[from] === true) {
                prevContentEnd = x + stringWidth(lines[from - 1]!)
              }

              lines = lines.slice(from, to)
              swFrom = from

              if (y < clip.y1!) {
                y = clip.y1!
              }
            }
          }

          const swBits = screen.softWrap
          let offsetY = 0

          for (const line of lines) {
            const lineY = y + offsetY

            // Line can be outside screen if `text` is taller than screen height
            if (lineY >= screenHeight) {
              break
            }

            const contentEnd = writeLineToScreen(screen, line, x, lineY, screenWidth, this.stylePool, this.charCache)

            writeCells += contentEnd - x

            // See Screen.softWrap docstring for the encoding. contentEnd
            // from writeLineToScreen is tab-expansion-aware, unlike
            // x+stringWidth(line) which treats tabs as width 0.
            if (softWrap) {
              const isSW = softWrap[swFrom + offsetY] === true
              swBits[lineY] = isSW ? prevContentEnd : 0
              prevContentEnd = contentEnd
            }

            offsetY++
          }

          continue
        }
      }
    }

    // noSelect ops go LAST so they win over blits (which copy noSelect
    // from prevScreen) and writes (which don't touch noSelect). This way
    // a <NoSelect> box correctly fences its region even when the parent
    // blits, and moving a <NoSelect> between frames correctly clears the
    // old region (resetScreen already zeroed the bitmap).
    for (const operation of this.operations) {
      if (operation.type === 'noSelect') {
        const { x, y, width, height } = operation.region
        markNoSelectRegion(screen, x, y, width, height)
      }
    }

    // Log blit/write ratio for debugging - high write count suggests blitting isn't working
    const totalCells = blitCells + writeCells

    if (totalCells > 1000 && writeCells > blitCells) {
      logForDebugging(
        `High write ratio: blit=${blitCells}, write=${writeCells} (${((writeCells / totalCells) * 100).toFixed(1)}% writes), screen=${screenHeight}x${screenWidth}`
      )
    }

    return screen
  }
}

function stylesEqual(a: AnsiCode[], b: AnsiCode[]): boolean {
  if (a === b) {
    return true
  } // Reference equality fast path

  const len = a.length

  if (len !== b.length) {
    return false
  }

  if (len === 0) {
    return true
  } // Both empty

  for (let i = 0; i < len; i++) {
    if (a[i]!.code !== b[i]!.code) {
      return false
    }
  }

  return true
}

/**
 * Convert a string with ANSI codes into styled characters with proper grapheme
 * clustering. Fixes ansi-tokenize splitting grapheme clusters (like family
 * emojis) into individual code points.
 *
 * Also precomputes styleId + hyperlink per style run (not per char) — an
 * 80-char line with 3 style runs does 3 intern calls instead of 80.
 */
function styledCharsWithGraphemeClustering(chars: StyledChar[], stylePool: StylePool): ClusteredChar[] {
  const charCount = chars.length

  if (charCount === 0) {
    return []
  }

  const result: ClusteredChar[] = []
  const bufferChars: string[] = []
  let bufferStyles: AnsiCode[] = chars[0]!.styles

  for (let i = 0; i < charCount; i++) {
    const char = chars[i]!
    const styles = char.styles

    // Different styles means we need to flush and start new buffer
    if (bufferChars.length > 0 && !stylesEqual(styles, bufferStyles)) {
      flushBuffer(bufferChars.join(''), bufferStyles, stylePool, result)
      bufferChars.length = 0
    }

    bufferChars.push(char.value)
    bufferStyles = styles
  }

  // Final flush
  if (bufferChars.length > 0) {
    flushBuffer(bufferChars.join(''), bufferStyles, stylePool, result)
  }

  return result
}

function flushBuffer(buffer: string, styles: AnsiCode[], stylePool: StylePool, out: ClusteredChar[]): void {
  // Compute styleId + hyperlink ONCE for the whole style run.
  // Every grapheme in this buffer shares the same styles.
  //
  // Extract and track hyperlinks separately, filter from styles.
  // Always check for OSC 8 codes to filter, not just when a URL is
  // extracted. The tokenizer treats OSC 8 close codes (empty URL) as
  // active styles, so they must be filtered even when no hyperlink
  // URL is present.
  const hyperlink = extractHyperlinkFromStyles(styles) ?? undefined

  const hasOsc8Styles =
    hyperlink !== undefined || styles.some(s => s.code.length >= OSC8_PREFIX.length && s.code.startsWith(OSC8_PREFIX))

  const filteredStyles = hasOsc8Styles ? filterOutHyperlinkStyles(styles) : styles

  const styleId = stylePool.intern(filteredStyles)

  for (const { segment: grapheme } of getGraphemeSegmenter().segment(buffer)) {
    out.push({
      value: grapheme,
      width: stringWidth(grapheme),
      styleId,
      hyperlink
    })
  }
}

/**
 * Write a single line's characters into the screen buffer.
 * Extracted from Output.get() so JSC can optimize this tight,
 * monomorphic loop independently — better register allocation,
 * setCellAt inlining, and type feedback than when buried inside
 * a 300-line dispatch function.
 *
 * Returns the end column (x + visual width, including tab expansion) so
 * the caller can record it in screen.softWrap without re-walking the
 * line via stringWidth(). Caller computes the debug cell-count as end-x.
 */
function writeLineToScreen(
  screen: Screen,
  line: string,
  x: number,
  y: number,
  screenWidth: number,
  stylePool: StylePool,
  charCache: Map<string, ClusteredChar[]>
): number {
  let characters = charCache.get(line)

  if (!characters) {
    characters = reorderBidi(styledCharsWithGraphemeClustering(styledCharsFromTokens(tokenize(line)), stylePool))
    charCache.set(line, characters)
  }

  let offsetX = x

  for (let charIdx = 0; charIdx < characters.length; charIdx++) {
    const character = characters[charIdx]!
    const codePoint = character.value.codePointAt(0)

    // Handle C0 control characters (0x00-0x1F) that cause cursor movement
    // mismatches. stringWidth treats these as width 0, but terminals may
    // move the cursor differently.
    if (codePoint !== undefined && codePoint <= 0x1f) {
      // Tab (0x09): expand to spaces to reach next tab stop
      if (codePoint === 0x09) {
        const tabWidth = 8
        const spacesToNextStop = tabWidth - (offsetX % tabWidth)

        for (let i = 0; i < spacesToNextStop && offsetX < screenWidth; i++) {
          setCellAt(screen, offsetX, y, {
            char: ' ',
            styleId: stylePool.none,
            width: CellWidth.Narrow,
            hyperlink: undefined
          })
          offsetX++
        }
      }
      // ESC (0x1B): skip incomplete escape sequences that ansi-tokenize
      // didn't recognize. ansi-tokenize only parses SGR sequences (ESC[...m)
      // and OSC 8 hyperlinks (ESC]8;;url BEL). Other sequences like cursor
      // movement, screen clearing, or terminal title become individual char
      // tokens that we need to skip here.
      else if (codePoint === 0x1b) {
        const nextChar = characters[charIdx + 1]?.value
        const nextCode = nextChar?.codePointAt(0)

        if (nextChar === '(' || nextChar === ')' || nextChar === '*' || nextChar === '+') {
          // Charset selection: ESC ( X, ESC ) X, etc.
          // Skip the intermediate char and the charset designator
          charIdx += 2
        } else if (nextChar === '[') {
          // CSI sequence: ESC [ ... final-byte
          // Final byte is in range 0x40-0x7E (@, A-Z, [\]^_`, a-z, {|}~)
          // Examples: ESC[2J (clear), ESC[?25l (cursor hide), ESC[H (home)
          charIdx++ // skip the [

          while (charIdx < characters.length - 1) {
            charIdx++
            const c = characters[charIdx]?.value.codePointAt(0)

            // Final byte terminates the sequence
            if (c !== undefined && c >= 0x40 && c <= 0x7e) {
              break
            }
          }
        } else if (nextChar === ']' || nextChar === 'P' || nextChar === '_' || nextChar === '^' || nextChar === 'X') {
          // String-based sequences terminated by BEL (0x07) or ST (ESC \):
          // - OSC: ESC ] ... (Operating System Command)
          // - DCS: ESC P ... (Device Control String)
          // - APC: ESC _ ... (Application Program Command)
          // - PM:  ESC ^ ... (Privacy Message)
          // - SOS: ESC X ... (Start of String)
          charIdx++ // skip the introducer char

          while (charIdx < characters.length - 1) {
            charIdx++
            const c = characters[charIdx]?.value

            // BEL (0x07) terminates the sequence
            if (c === '\x07') {
              break
            }

            // ST (String Terminator) is ESC \
            // When we see ESC, check if next char is backslash
            if (c === '\x1b') {
              const nextC = characters[charIdx + 1]?.value

              if (nextC === '\\') {
                charIdx++ // skip the backslash too

                break
              }
            }
          }
        } else if (nextCode !== undefined && nextCode >= 0x30 && nextCode <= 0x7e) {
          // Single-character escape sequences: ESC followed by 0x30-0x7E
          // (excluding the multi-char introducers already handled above)
          // - Fp range (0x30-0x3F): ESC 7 (save cursor), ESC 8 (restore)
          // - Fe range (0x40-0x5F): ESC D (index), ESC M (reverse index)
          // - Fs range (0x60-0x7E): ESC c (reset)
          charIdx++ // skip the command char
        }
      }

      // Carriage return (0x0D): would move cursor to column 0, skip it
      // Backspace (0x08): would move cursor left, skip it
      // Bell (0x07), vertical tab (0x0B), form feed (0x0C): skip
      // All other control chars (0x00-0x06, 0x0E-0x1F): skip
      // Note: newline (0x0A) is already handled by line splitting
      continue
    }

    // Zero-width characters (combining marks, ZWNJ, ZWS, etc.)
    // don't occupy terminal cells — storing them as Narrow cells
    // desyncs the virtual cursor from the real terminal cursor.
    // Width was computed once during clustering (cached via charCache).
    const charWidth = character.width

    if (charWidth === 0) {
      continue
    }

    const isWideCharacter = charWidth >= 2

    // Wide char at last column can't fit — terminal would wrap it to
    // the next line, desyncing our cursor model. Place a SpacerHead
    // to mark the blank column, matching terminal behavior.
    if (isWideCharacter && offsetX + 2 > screenWidth) {
      setCellAt(screen, offsetX, y, {
        char: ' ',
        styleId: stylePool.none,
        width: CellWidth.SpacerHead,
        hyperlink: undefined
      })
      offsetX++

      continue
    }

    // styleId + hyperlink were precomputed during clustering (once per
    // style run, cached via charCache). Hot loop is now just property
    // reads — no intern, no extract, no filter per frame.
    setCellAt(screen, offsetX, y, {
      char: character.value,
      styleId: character.styleId,
      width: isWideCharacter ? CellWidth.Wide : CellWidth.Narrow,
      hyperlink: character.hyperlink
    })
    offsetX += isWideCharacter ? 2 : 1
  }

  return offsetX
}
