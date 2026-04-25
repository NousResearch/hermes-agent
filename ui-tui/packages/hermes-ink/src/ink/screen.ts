import { ansiCodesToString, diffAnsiCodes, type AnsiCode } from '@alcalzone/ansi-tokenize'

import { unionRect, type Point, type Rectangle, type Size } from './layout/geometry.js'
import { BEL, ESC, SEP } from './termio/ansi.js'
import * as warn from './warn.js'

// --- Shared Pools (interning for memory efficiency) ---

// Character string pool shared across all screens.
// With a shared pool, interned char IDs are valid across screens,
// so blitRegion can copy IDs directly (no re-interning) and
// diffEach can compare IDs as integers (no string lookup).
export class CharPool {
  private strings: string[] = [' ', ''] // Index 0 = space, 1 = empty (spacer)
  private stringMap = new Map<string, number>([
    [' ', 0],
    ['', 1]
  ])
  private ascii: Int32Array = initCharAscii() // charCode → index, -1 = not interned

  intern(char: string): number {
    // ASCII fast-path: direct array lookup instead of Map.get
    if (char.length === 1) {
      const code = char.charCodeAt(0)

      if (code < 128) {
        const cached = this.ascii[code]!

        if (cached !== -1) {
          return cached
        }

        const index = this.strings.length
        this.strings.push(char)
        this.ascii[code] = index

        return index
      }
    }

    const existing = this.stringMap.get(char)

    if (existing !== undefined) {
      return existing
    }

    const index = this.strings.length
    this.strings.push(char)
    this.stringMap.set(char, index)

    return index
  }

  get(index: number): string {
    return this.strings[index] ?? ' '
  }
}

// Hyperlink string pool shared across all screens.
// Index 0 = no hyperlink.
export class HyperlinkPool {
  private strings: string[] = [''] // Index 0 = no hyperlink
  private stringMap = new Map<string, number>()

  intern(hyperlink: string | undefined): number {
    if (!hyperlink) {
      return 0
    }

    let id = this.stringMap.get(hyperlink)

    if (id === undefined) {
      id = this.strings.length
      this.strings.push(hyperlink)
      this.stringMap.set(hyperlink, id)
    }

    return id
  }

  get(id: number): string | undefined {
    return id === 0 ? undefined : this.strings[id]
  }
}

// SGR 7 (inverse) as an AnsiCode. endCode '\x1b[27m' flags VISIBLE_ON_SPACE
// so bit 0 of the resulting styleId is set → renderer won't skip inverted
// spaces as invisible.
const INVERSE_CODE: AnsiCode = {
  type: 'ansi',
  code: '\x1b[7m',
  endCode: '\x1b[27m'
}

// Bold (SGR 1) — stacks cleanly, no reflow in monospace. endCode 22
// also cancels dim (SGR 2); harmless here since we never add dim.
const BOLD_CODE: AnsiCode = {
  type: 'ansi',
  code: '\x1b[1m',
  endCode: '\x1b[22m'
}

// Underline (SGR 4). Kept alongside yellow+bold — the underline is the
// unambiguous visible-on-any-theme marker. Yellow-bg-via-inverse can
// clash with existing bg colors (user-prompt style, tool chrome, syntax
// bg). If you see underline but no yellow, the yellow is being lost in
// the existing cell styling — the overlay IS finding the match.
const UNDERLINE_CODE: AnsiCode = {
  type: 'ansi',
  code: '\x1b[4m',
  endCode: '\x1b[24m'
}

// fg→yellow (SGR 33). With inverse already in the stack, the terminal
// swaps fg↔bg at render — so yellow-fg becomes yellow-BG. Original bg
// becomes fg (readable on most themes: dark-bg → dark-text on yellow).
// endCode 39 is 'default fg' — cancels any prior fg color cleanly.
const YELLOW_FG_CODE: AnsiCode = {
  type: 'ansi',
  code: '\x1b[33m',
  endCode: '\x1b[39m'
}

const MAX_TRANSITION_CACHE = 32768

export class StylePool {
  private ids = new Map<string, number>()
  private styles: AnsiCode[][] = []
  private transitionCache = new Map<number, string>()
  readonly none: number

  constructor() {
    this.none = this.intern([])
  }

  /**
   * Intern a style and return its ID. Bit 0 of the ID encodes whether the
   * style has a visible effect on space characters (background, inverse,
   * underline, etc.). Foreground-only styles get even IDs; styles visible
   * on spaces get odd IDs. This lets the renderer skip invisible spaces
   * with a single bitmask check on the packed word.
   */
  intern(styles: AnsiCode[]): number {
    const key = styles.length === 0 ? '' : styles.map(s => s.code).join('\0')
    let id = this.ids.get(key)

    if (id === undefined) {
      const rawId = this.styles.length
      this.styles.push(styles.length === 0 ? [] : styles)
      id = (rawId << 1) | (styles.length > 0 && hasVisibleSpaceEffect(styles) ? 1 : 0)
      this.ids.set(key, id)
    }

    return id
  }

  /** Recover styles from an encoded ID. Strips the bit-0 flag via >>> 1. */
  get(id: number): AnsiCode[] {
    return this.styles[id >>> 1] ?? []
  }

  /**
   * Returns the pre-serialized ANSI string to transition from one style to
   * another. Cached by (fromId, toId) — zero allocations after first call
   * for a given pair. Full-clear at MAX_TRANSITION_CACHE guards against
   * unbounded growth from ever-expanding id spaces; cache repopulates from
   * the next frame's actual transitions.
   */
  transition(fromId: number, toId: number): string {
    if (fromId === toId) {
      return ''
    }

    const key = fromId * 0x100000 + toId
    let str = this.transitionCache.get(key)

    if (str === undefined) {
      if (this.transitionCache.size >= MAX_TRANSITION_CACHE) {
        this.transitionCache.clear()
      }

      str = ansiCodesToString(diffAnsiCodes(this.get(fromId), this.get(toId)))
      this.transitionCache.set(key, str)
    }

    return str
  }

  /**
   * Intern a style that is `base + inverse`. Cached by base ID so
   * repeated calls for the same underlying style don't re-scan the
   * AnsiCode[] array. Used by the selection overlay.
   */
  private inverseCache = new Map<number, number>()
  withInverse(baseId: number): number {
    let id = this.inverseCache.get(baseId)

    if (id === undefined) {
      const baseCodes = this.get(baseId)
      // If already inverted, use as-is (avoids SGR 7 stacking)
      const hasInverse = baseCodes.some(c => c.endCode === '\x1b[27m')
      id = hasInverse ? baseId : this.intern([...baseCodes, INVERSE_CODE])
      this.inverseCache.set(baseId, id)
    }

    return id
  }

  /** Inverse + bold + yellow-bg-via-fg-swap for the CURRENT search match.
   *  OTHER matches are plain inverse — bg inherits from the theme. Current
   *  gets a distinct yellow bg (via fg-then-inverse swap) plus bold weight
   *  so it stands out in a sea of inverse. Underline was too subtle. Zero
   *  reflow risk: all pure SGR overlays, per-cell, post-layout. The yellow
   *  overrides any existing fg (syntax highlighting) on those cells — fine,
   *  the "you are here" signal IS the point, syntax color can yield. */
  private currentMatchCache = new Map<number, number>()
  withCurrentMatch(baseId: number): number {
    let id = this.currentMatchCache.get(baseId)

    if (id === undefined) {
      const baseCodes = this.get(baseId)

      // Filter BOTH fg + bg so yellow-via-inverse is unambiguous.
      // User-prompt cells have an explicit bg (grey box); with that bg
      // still set, inverse swaps yellow-fg↔grey-bg → grey-on-yellow on
      // SOME terminals, yellow-on-grey on others (inverse semantics vary
      // when both colors are explicit). Filtering both gives clean
      // yellow-bg + terminal-default-fg everywhere. Bold/dim/italic
      // coexist — keep those.
      const codes = baseCodes.filter(c => c.endCode !== '\x1b[39m' && c.endCode !== '\x1b[49m')

      // fg-yellow FIRST so inverse swaps it to bg. Bold after inverse is
      // fine — SGR 1 is fg-attribute-only, order-independent vs 7.
      codes.push(YELLOW_FG_CODE)

      if (!baseCodes.some(c => c.endCode === '\x1b[27m')) {
        codes.push(INVERSE_CODE)
      }

      if (!baseCodes.some(c => c.endCode === '\x1b[22m')) {
        codes.push(BOLD_CODE)
      }

      // Underline as the unambiguous marker — yellow-bg can clash with
      // existing bg styling (user-prompt bg, syntax bg). If you see
      // underline but no yellow on a match, the overlay IS finding it;
      // the yellow is just losing a styling fight.
      if (!baseCodes.some(c => c.endCode === '\x1b[24m')) {
        codes.push(UNDERLINE_CODE)
      }

      id = this.intern(codes)
      this.currentMatchCache.set(baseId, id)
    }

    return id
  }

  /**
   * Selection overlay: REPLACE the cell's background with a solid color
   * while preserving its foreground (color, bold, italic, dim, underline).
   * Matches native terminal selection — a dedicated bg color, not SGR-7
   * inverse. Inverse swaps fg/bg per-cell, which fragments visually over
   * syntax-highlighted text (every fg color becomes a different bg stripe).
   *
   * Strips any existing bg (endCode 49m — REPLACES, so diff-added green
   * etc. don't bleed through) and any existing inverse (endCode 27m —
   * inverse on top of a solid bg would re-swap and look wrong).
   *
   * bg is set via setSelectionBg(); null → fallback to withInverse() so the
   * overlay still works before theme wiring sets a color (tests, first frame).
   * Cache is keyed by baseId only — setSelectionBg() clears it on change.
   */
  private selectionBgCode: AnsiCode | null = null
  private selectionBgCache = new Map<number, number>()
  setSelectionBg(bg: AnsiCode | null): void {
    if (this.selectionBgCode?.code === bg?.code) {
      return
    }

    this.selectionBgCode = bg
    this.selectionBgCache.clear()
  }
  withSelectionBg(baseId: number): number {
    const bg = this.selectionBgCode

    if (bg === null) {
      return this.withInverse(baseId)
    }

    let id = this.selectionBgCache.get(baseId)

    if (id === undefined) {
      // Keep everything except bg (49m) and inverse (27m). Fg, bold, dim,
      // italic, underline, strikethrough all preserved.
      const kept = this.get(baseId).filter(c => c.endCode !== '\x1b[49m' && c.endCode !== '\x1b[27m')

      kept.push(bg)
      id = this.intern(kept)
      this.selectionBgCache.set(baseId, id)
    }

    return id
  }
}

// endCodes that produce visible effects on space characters
const VISIBLE_ON_SPACE = new Set([
  '\x1b[49m', // background color
  '\x1b[27m', // inverse
  '\x1b[24m', // underline
  '\x1b[29m', // strikethrough
  '\x1b[55m' // overline
])

function hasVisibleSpaceEffect(styles: AnsiCode[]): boolean {
  for (const style of styles) {
    if (VISIBLE_ON_SPACE.has(style.endCode)) {
      return true
    }
  }

  return false
}

/**
 * Cell width classification for handling double-wide characters (CJK, emoji,
 * etc.)
 *
 * We use explicit spacer cells rather than inferring width at render time. This
 * makes the data structure self-describing and simplifies cursor positioning
 * logic.
 *
 * @see https://mitchellh.com/writing/grapheme-clusters-in-terminals
 */
// const enum is inlined at compile time - no runtime object, no property access
export const enum CellWidth {
  // Not a wide character, cell width 1
  Narrow = 0,
  // Wide character, cell width 2. This cell contains the actual character.
  Wide = 1,
  // Spacer occupying the second visual column of a wide character. Do not render.
  SpacerTail = 2,
  // Spacer at the end of a soft-wrapped line indicating that a wide character
  // continues on the next line. Used for preserving wide character semantics
  // across line breaks during soft wrapping.
  SpacerHead = 3
}

export type Hyperlink = string | undefined

/**
 * Cell is a view type returned by cellAt(). Cells are stored as packed typed
 * arrays internally to avoid GC pressure from allocating objects per cell.
 */
export type Cell = {
  char: string
  styleId: number
  width: CellWidth
  hyperlink: Hyperlink
}

// Constants for empty/spacer cells to enable fast comparisons
// These are indices into the charStrings table, not codepoints
const EMPTY_CHAR_INDEX = 0 // ' ' (space)
const SPACER_CHAR_INDEX = 1 // '' (empty string for spacer cells)
// Unwritten cells are [EMPTY_CHAR_INDEX=0, packWord1(emptyStyleId=0,0,0)=0].
// Since StylePool.none is always 0 (first intern), unwritten cells are
// indistinguishable from explicitly-cleared cells in the packed array.
// This is intentional: diffEach can compare raw ints with zero normalization.
// isEmptyCellByIndex checks if both words are 0 to identify "never visually written" cells.

function initCharAscii(): Int32Array {
  const table = new Int32Array(128)
  table.fill(-1)
  table[32] = EMPTY_CHAR_INDEX // ' ' (space)

  return table
}

// --- Packed cell layout ---
// Each cell is 2 consecutive Int32 elements in the cells array:
//   word0 (cells[ci]):     charId (full 32 bits)
//   word1 (cells[ci + 1]): styleId[31:17] | hyperlinkId[16:2] | width[1:0]
const STYLE_SHIFT = 17
const HYPERLINK_SHIFT = 2
const HYPERLINK_MASK = 0x7fff // 15 bits
const WIDTH_MASK = 3 // 2 bits

// Pack styleId, hyperlinkId, and width into a single Int32
function packWord1(styleId: number, hyperlinkId: number, width: number): number {
  return (styleId << STYLE_SHIFT) | (hyperlinkId << HYPERLINK_SHIFT) | width
}

// Unwritten cell as BigInt64 — both words are 0, so the 64-bit value is 0n.
// Used by BigInt64Array.fill() for bulk clears (resetScreen, clearRegion).
// Not used for comparison — BigInt element reads cause heap allocation.
const EMPTY_CELL_VALUE = 0n

/**
 * Screen uses a packed Int32Array instead of Cell objects to eliminate GC
 * pressure. For a 200x120 screen, this avoids allocating 24,000 objects.
 *
 * Cell data is stored as 2 Int32s per cell in a single contiguous array:
 *   word0: charId (full 32 bits — index into CharPool)
 *   word1: styleId[31:17] | hyperlinkId[16:2] | width[1:0]
 *
 * This layout halves memory accesses in diffEach (2 int loads vs 4) and
 * enables future SIMD comparison via Bun.indexOfFirstDifference.
 */
export type Screen = Size & {
  // Packed cell data — 2 Int32s per cell: [charId, packed(styleId|hyperlinkId|width)]
  // cells and cells64 are views over the same ArrayBuffer.
  cells: Int32Array
  cells64: BigInt64Array // 1 BigInt64 per cell — used for bulk fill in resetScreen/clearRegion

  // Shared pools — IDs are valid across all screens using the same pools
  charPool: CharPool
  hyperlinkPool: HyperlinkPool

  // Empty style ID for comparisons
  emptyStyleId: number

  /**
   * Bounding box of cells that were written to (not blitted) during rendering.
   * Used by diff() to limit iteration to only the region that could have changed.
   */
  damage: Rectangle | undefined

  /**
   * Per-cell noSelect bitmap — 1 byte per cell, 1 = exclude from text
   * selection (copy + highlight). Used by <NoSelect> to mark gutters
   * (line numbers, diff sigils) so click-drag over a diff yields clean
   * copyable code. Fully reset each frame in resetScreen; blitRegion
   * copies it alongside cells so the blit optimization preserves marks.
   */
  noSelect: Uint8Array

  /**
   * Per-cell written bitmap. A written plain space and never-written padding
   * share the same packed cell value, so selection needs this side channel to
   * preserve code indentation without selecting blank UI margins.
   */
  written: Uint8Array

  /**
   * Per-ROW soft-wrap continuation marker. softWrap[r]=N>0 means row r
   * is a word-wrap continuation of row r-1 (the `\n` before it was
   * inserted by wrapAnsi, not in the source), and row r-1's written
   * content ends at absolute column N (exclusive — cells [0..N) are the
   * fragment, past N is unwritten padding). 0 means row r is NOT a
   * continuation (hard newline or first row). Selection copy checks
   * softWrap[r]>0 to join row r onto row r-1 without a newline, and
   * reads softWrap[r+1] to know row r's content end when row r+1
   * continues from it. The content-end column is needed because an
   * unwritten cell and a written-unstyled-space are indistinguishable in
   * the packed typed array (both all-zero) — without it we'd either drop
   * the word-separator space (trim) or include trailing padding (no
   * trim). This encoding (continuation-on-self, prev-content-end-here)
   * is chosen so shiftRows preserves the is-continuation semantics: when
   * row r scrolls off the top and row r+1 shifts to row r, sw[r] gets
   * old sw[r+1] — which correctly says the new row r is a continuation
   * of what's now in scrolledOffAbove. Reset each frame; copied by
   * blitRegion/shiftRows.
   */
  softWrap: Int32Array
}

function isEmptyCellByIndex(screen: Screen, index: number): boolean {
  // An empty/unwritten cell has both words === 0:
  // word0 = EMPTY_CHAR_INDEX (0), word1 = packWord1(emptyStyleId=0, 0, 0) = 0.
  const ci = index << 1

  return screen.cells[ci] === 0 && screen.cells[ci | 1] === 0
}

export function isEmptyCellAt(screen: Screen, x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return true
  }

  return isEmptyCellByIndex(screen, y * screen.width + x)
}

export function isWrittenCellAt(screen: Screen, x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return false
  }

  return screen.written[y * screen.width + x] === 1
}

/**
 * Check if a Cell (view object) represents an empty cell.
 */
export function isCellEmpty(screen: Screen, cell: Cell): boolean {
  // Check if cell looks like an empty cell (space, empty style, narrow, no link).
  // Note: After cellAt mapping, unwritten cells have emptyStyleId, so this
  // returns true for both unwritten AND cleared cells. Use isEmptyCellAt
  // for the internal distinction.
  return cell.char === ' ' && cell.styleId === screen.emptyStyleId && cell.width === CellWidth.Narrow && !cell.hyperlink
}

// Intern a hyperlink string and return its ID (0 = no hyperlink)
function internHyperlink(screen: Screen, hyperlink: Hyperlink): number {
  return screen.hyperlinkPool.intern(hyperlink)
}

// ---

export function createScreen(
  width: number,
  height: number,
  styles: StylePool,
  charPool: CharPool,
  hyperlinkPool: HyperlinkPool
): Screen {
  // Warn if dimensions are not valid integers (likely bad yoga layout output)
  warn.ifNotInteger(width, 'createScreen width')
  warn.ifNotInteger(height, 'createScreen height')

  // Ensure width and height are valid integers to prevent crashes
  if (!Number.isInteger(width) || width < 0) {
    width = Math.max(0, Math.floor(width) || 0)
  }

  if (!Number.isInteger(height) || height < 0) {
    height = Math.max(0, Math.floor(height) || 0)
  }

  const size = width * height

  // Allocate one buffer, two views: Int32Array for per-word access,
  // BigInt64Array for bulk fill in resetScreen/clearRegion.
  // ArrayBuffer is zero-filled, which is exactly the empty cell value:
  // [EMPTY_CHAR_INDEX=0, packWord1(emptyStyleId=0,0,0)=0].
  const buf = new ArrayBuffer(size << 3) // 8 bytes per cell
  const cells = new Int32Array(buf)
  const cells64 = new BigInt64Array(buf)

  return {
    width,
    height,
    cells,
    cells64,
    charPool,
    hyperlinkPool,
    emptyStyleId: styles.none,
    damage: undefined,
    noSelect: new Uint8Array(size),
    written: new Uint8Array(size),
    softWrap: new Int32Array(height)
  }
}

/**
 * Reset an existing screen for reuse, avoiding allocation of new typed arrays.
 * Resizes if needed and clears all cells to empty/unwritten state.
 *
 * For double-buffering, this allows swapping between front and back buffers
 * without allocating new Screen objects each frame.
 */
export function resetScreen(screen: Screen, width: number, height: number): void {
  // Warn if dimensions are not valid integers
  warn.ifNotInteger(width, 'resetScreen width')
  warn.ifNotInteger(height, 'resetScreen height')

  // Ensure width and height are valid integers to prevent crashes
  if (!Number.isInteger(width) || width < 0) {
    width = Math.max(0, Math.floor(width) || 0)
  }

  if (!Number.isInteger(height) || height < 0) {
    height = Math.max(0, Math.floor(height) || 0)
  }

  const size = width * height

  // Resize if needed (only grow, to avoid reallocations)
  if (screen.cells64.length < size) {
    const buf = new ArrayBuffer(size << 3)
    screen.cells = new Int32Array(buf)
    screen.cells64 = new BigInt64Array(buf)
    screen.noSelect = new Uint8Array(size)
    screen.written = new Uint8Array(size)
  }

  if (screen.softWrap.length < height) {
    screen.softWrap = new Int32Array(height)
  }

  // Reset all cells — single fill call, no loop
  screen.cells64.fill(EMPTY_CELL_VALUE, 0, size)
  screen.noSelect.fill(0, 0, size)
  screen.written.fill(0, 0, size)
  screen.softWrap.fill(0, 0, height)

  // Update dimensions
  screen.width = width
  screen.height = height

  // Shared pools accumulate — no clearing needed. Unique char/hyperlink sets are bounded.

  // Clear damage tracking
  screen.damage = undefined
}

/**
 * Re-intern a screen's char and hyperlink IDs into new pools.
 * Used for generational pool reset — after migrating, the screen's
 * typed arrays contain valid IDs for the new pools, and the old pools
 * can be GC'd.
 *
 * O(width * height) but only called occasionally (e.g., between conversation turns).
 */
export function migrateScreenPools(screen: Screen, charPool: CharPool, hyperlinkPool: HyperlinkPool): void {
  const oldCharPool = screen.charPool
  const oldHyperlinkPool = screen.hyperlinkPool

  if (oldCharPool === charPool && oldHyperlinkPool === hyperlinkPool) {
    return
  }

  const size = screen.width * screen.height
  const cells = screen.cells

  // Re-intern chars and hyperlinks in a single pass, stride by 2
  for (let ci = 0; ci < size << 1; ci += 2) {
    // Re-intern charId (word0)
    const oldCharId = cells[ci]!
    cells[ci] = charPool.intern(oldCharPool.get(oldCharId))

    // Re-intern hyperlinkId (packed in word1)
    const word1 = cells[ci + 1]!
    const oldHyperlinkId = (word1 >>> HYPERLINK_SHIFT) & HYPERLINK_MASK

    if (oldHyperlinkId !== 0) {
      const oldStr = oldHyperlinkPool.get(oldHyperlinkId)
      const newHyperlinkId = hyperlinkPool.intern(oldStr)
      // Repack word1 with new hyperlinkId, preserving styleId and width
      const styleId = word1 >>> STYLE_SHIFT
      const width = word1 & WIDTH_MASK
      cells[ci + 1] = packWord1(styleId, newHyperlinkId, width)
    }
  }

  screen.charPool = charPool
  screen.hyperlinkPool = hyperlinkPool
}

/**
 * Get a Cell view at the given position. Returns a new object each call -
 * this is intentional as cells are stored packed, not as objects.
 */
export function cellAt(screen: Screen, x: number, y: number): Cell | undefined {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return undefined
  }

  return cellAtIndex(screen, y * screen.width + x)
}

/**
 * Get a Cell view by pre-computed array index. Skips bounds checks and
 * index computation — caller must ensure index is valid.
 */
export function cellAtIndex(screen: Screen, index: number): Cell {
  const ci = index << 1
  const word1 = screen.cells[ci + 1]!
  const hid = (word1 >>> HYPERLINK_SHIFT) & HYPERLINK_MASK

  return {
    // Unwritten cells have charIndex=0 (EMPTY_CHAR_INDEX); charPool.get(0) returns ' '
    char: screen.charPool.get(screen.cells[ci]!),
    styleId: word1 >>> STYLE_SHIFT,
    width: word1 & WIDTH_MASK,
    hyperlink: hid === 0 ? undefined : screen.hyperlinkPool.get(hid)
  }
}

/**
 * Get a Cell at the given index, or undefined if it has no visible content.
 * Returns undefined for spacer cells (charId 1), empty unstyled spaces, and
 * fg-only styled spaces that match lastRenderedStyleId (cursor-forward
 * produces an identical visual result, avoiding a Cell allocation).
 *
 * @param lastRenderedStyleId - styleId of the last rendered cell on this
 *   line, or -1 if none yet.
 */
export function visibleCellAtIndex(
  cells: Int32Array,
  charPool: CharPool,
  hyperlinkPool: HyperlinkPool,
  index: number,
  lastRenderedStyleId: number
): Cell | undefined {
  const ci = index << 1
  const charId = cells[ci]!

  if (charId === 1) {
    return undefined
  } // spacer

  const word1 = cells[ci + 1]!

  // For spaces: 0x3fffc masks bits 2-17 (hyperlinkId + styleId visibility
  // bit). If zero, the space has no hyperlink and at most a fg-only style.
  // Then word1 >>> STYLE_SHIFT is the foreground style — skip if it's zero
  // (truly invisible) or matches the last rendered style on this line.
  if (charId === 0 && (word1 & 0x3fffc) === 0) {
    const fgStyle = word1 >>> STYLE_SHIFT

    if (fgStyle === 0 || fgStyle === lastRenderedStyleId) {
      return undefined
    }
  }

  const hid = (word1 >>> HYPERLINK_SHIFT) & HYPERLINK_MASK

  return {
    char: charPool.get(charId),
    styleId: word1 >>> STYLE_SHIFT,
    width: word1 & WIDTH_MASK,
    hyperlink: hid === 0 ? undefined : hyperlinkPool.get(hid)
  }
}

/**
 * Write cell data into an existing Cell object to avoid allocation.
 * Caller must ensure index is valid.
 */
function cellAtCI(screen: Screen, ci: number, out: Cell): void {
  const w1 = ci | 1
  const word1 = screen.cells[w1]!
  out.char = screen.charPool.get(screen.cells[ci]!)
  out.styleId = word1 >>> STYLE_SHIFT
  out.width = word1 & WIDTH_MASK
  const hid = (word1 >>> HYPERLINK_SHIFT) & HYPERLINK_MASK
  out.hyperlink = hid === 0 ? undefined : screen.hyperlinkPool.get(hid)
}

export function charInCellAt(screen: Screen, x: number, y: number): string | undefined {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return undefined
  }

  const ci = (y * screen.width + x) << 1

  return screen.charPool.get(screen.cells[ci]!)
}

/**
 * Set a cell, optionally creating a spacer for wide characters.
 *
 * Wide characters (CJK, emoji) occupy 2 cells in the buffer:
 * 1. First cell: Contains the actual character with width = Wide
 * 2. Second cell: Spacer cell with width = SpacerTail (empty, not rendered)
 *
 * If the cell has width = Wide, this function automatically creates the
 * corresponding SpacerTail in the next column. This two-cell model keeps
 * the buffer aligned to visual columns, making cursor positioning
 * straightforward.
 *
 * TODO: When soft-wrapping is implemented, SpacerHead cells will be explicitly
 * placed by the wrapping logic at line-end positions where wide characters
 * wrap to the next line. This function doesn't need to handle SpacerHead
 * automatically - it will be set directly by the wrapping code.
 */
export function setCellAt(screen: Screen, x: number, y: number, cell: Cell): void {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return
  }

  const ci = (y * screen.width + x) << 1
  const cells = screen.cells

  // When a Wide char is overwritten by a Narrow char, its SpacerTail remains
  // as a ghost cell that the diff/render pipeline skips, causing stale content
  // to leak through from previous frames.
  const prevWidth = cells[ci + 1]! & WIDTH_MASK

  if (prevWidth === CellWidth.Wide && cell.width !== CellWidth.Wide) {
    const spacerX = x + 1

    if (spacerX < screen.width) {
      const spacerCI = ci + 2

      if ((cells[spacerCI + 1]! & WIDTH_MASK) === CellWidth.SpacerTail) {
        cells[spacerCI] = EMPTY_CHAR_INDEX
        cells[spacerCI + 1] = packWord1(screen.emptyStyleId, 0, CellWidth.Narrow)
        screen.written[y * screen.width + spacerX] = 0
      }
    }
  }

  // Track cleared Wide position for damage expansion below
  let clearedWideX = -1

  if (prevWidth === CellWidth.SpacerTail && cell.width !== CellWidth.SpacerTail) {
    // Overwriting a SpacerTail: clear the orphaned Wide char at (x-1).
    // Keeping the wide character with Narrow width would cause the terminal
    // to still render it with width 2, desyncing the cursor model.
    if (x > 0) {
      const wideCI = ci - 2

      if ((cells[wideCI + 1]! & WIDTH_MASK) === CellWidth.Wide) {
        cells[wideCI] = EMPTY_CHAR_INDEX
        cells[wideCI + 1] = packWord1(screen.emptyStyleId, 0, CellWidth.Narrow)
        screen.written[y * screen.width + x - 1] = 0
        clearedWideX = x - 1
      }
    }
  }

  // Pack cell data into cells array
  cells[ci] = internCharString(screen, cell.char)
  cells[ci + 1] = packWord1(cell.styleId, internHyperlink(screen, cell.hyperlink), cell.width)
  screen.written[y * screen.width + x] = 1

  // Track damage - expand bounds in place instead of allocating new objects
  // Include the main cell position and any cleared orphan cells
  const minX = clearedWideX >= 0 ? Math.min(x, clearedWideX) : x
  const damage = screen.damage

  if (damage) {
    const right = damage.x + damage.width
    const bottom = damage.y + damage.height

    if (minX < damage.x) {
      damage.width += damage.x - minX
      damage.x = minX
    } else if (x >= right) {
      damage.width = x - damage.x + 1
    }

    if (y < damage.y) {
      damage.height += damage.y - y
      damage.y = y
    } else if (y >= bottom) {
      damage.height = y - damage.y + 1
    }
  } else {
    screen.damage = { x: minX, y, width: x - minX + 1, height: 1 }
  }

  // If this is a wide character, create a spacer in the next column
  if (cell.width === CellWidth.Wide) {
    const spacerX = x + 1

    if (spacerX < screen.width) {
      const spacerCI = ci + 2

      // If the cell we're overwriting with our SpacerTail is itself Wide,
      // clear ITS SpacerTail at x+2 too. Otherwise the orphan SpacerTail
      // makes diffEach report it as `added` and log-update's skip-spacer
      // rule prevents clearing whatever prev content was at that column.
      // Scenario: [a, 💻, spacer] → [本, spacer, ORPHAN spacer] when
      // yoga squishes a💻 to height 0 and 本 renders at the same y.
      if ((cells[spacerCI + 1]! & WIDTH_MASK) === CellWidth.Wide) {
        const orphanCI = spacerCI + 2

        if (spacerX + 1 < screen.width && (cells[orphanCI + 1]! & WIDTH_MASK) === CellWidth.SpacerTail) {
          cells[orphanCI] = EMPTY_CHAR_INDEX
          cells[orphanCI + 1] = packWord1(screen.emptyStyleId, 0, CellWidth.Narrow)
          screen.written[y * screen.width + spacerX + 1] = 0
        }
      }

      cells[spacerCI] = SPACER_CHAR_INDEX
      cells[spacerCI + 1] = packWord1(screen.emptyStyleId, 0, CellWidth.SpacerTail)
      screen.written[y * screen.width + spacerX] = 1

      // Expand damage to include SpacerTail so diff() scans it
      const d = screen.damage

      if (d && spacerX >= d.x + d.width) {
        d.width = spacerX - d.x + 1
      }
    }
  }
}

/**
 * Replace the styleId of a cell in-place without disturbing char, width,
 * or hyperlink. Preserves empty cells as-is (char stays ' '). Tracks damage
 * for the cell so diffEach picks up the change.
 */
export function setCellStyleId(screen: Screen, x: number, y: number, styleId: number): void {
  if (x < 0 || y < 0 || x >= screen.width || y >= screen.height) {
    return
  }

  const ci = (y * screen.width + x) << 1
  const cells = screen.cells
  const word1 = cells[ci + 1]!
  const width = word1 & WIDTH_MASK

  // Skip spacer cells — inverse on the head cell visually covers both columns
  if (width === CellWidth.SpacerTail || width === CellWidth.SpacerHead) {
    return
  }

  const hid = (word1 >>> HYPERLINK_SHIFT) & HYPERLINK_MASK
  cells[ci + 1] = packWord1(styleId, hid, width)
  // Expand damage so diffEach scans this cell
  const d = screen.damage

  if (d) {
    screen.damage = unionRect(d, { x, y, width: 1, height: 1 })
  } else {
    screen.damage = { x, y, width: 1, height: 1 }
  }
}

/**
 * Intern a character string via the screen's shared CharPool.
 * Supports grapheme clusters like family emoji.
 */
function internCharString(screen: Screen, char: string): number {
  return screen.charPool.intern(char)
}

/**
 * Bulk-copy a rectangular region from src to dst using TypedArray.set().
 * Single cells.set() call per row (or one call for contiguous blocks).
 * Damage is computed once for the whole region.
 *
 * Clamps negative regionX/regionY to 0 (matching clearRegion) — absolute-
 * positioned overlays in tiny terminals can compute negative screen coords.
 * maxX/maxY should already be clamped to both screen bounds by the caller.
 */
export function blitRegion(
  dst: Screen,
  src: Screen,
  regionX: number,
  regionY: number,
  maxX: number,
  maxY: number
): void {
  regionX = Math.max(0, regionX)
  regionY = Math.max(0, regionY)

  if (regionX >= maxX || regionY >= maxY) {
    return
  }

  const rowLen = maxX - regionX
  const srcStride = src.width << 1
  const dstStride = dst.width << 1
  const rowBytes = rowLen << 1 // 2 Int32s per cell
  const srcCells = src.cells
  const dstCells = dst.cells
  const srcNoSel = src.noSelect
  const dstNoSel = dst.noSelect
  const srcWritten = src.written
  const dstWritten = dst.written

  // softWrap is per-row — copy the row range regardless of stride/width.
  // Partial-width blits still carry the row's wrap provenance since the
  // blitted content (a cached ink-text node) is what set the bit.
  dst.softWrap.set(src.softWrap.subarray(regionY, maxY), regionY)

  // Fast path: contiguous memory when copying full-width rows at same stride
  if (regionX === 0 && maxX === src.width && src.width === dst.width) {
    const srcStart = regionY * srcStride
    const totalBytes = (maxY - regionY) * srcStride
    dstCells.set(
      srcCells.subarray(srcStart, srcStart + totalBytes),
      srcStart // srcStart === dstStart when strides match and regionX === 0
    )
    // noSelect is 1 byte/cell vs cells' 8 — same region, different scale
    const nsStart = regionY * src.width
    const nsLen = (maxY - regionY) * src.width
    dstNoSel.set(srcNoSel.subarray(nsStart, nsStart + nsLen), nsStart)
    dstWritten.set(srcWritten.subarray(nsStart, nsStart + nsLen), nsStart)
  } else {
    // Per-row copy for partial-width or mismatched-stride regions
    let srcRowCI = regionY * srcStride + (regionX << 1)
    let dstRowCI = regionY * dstStride + (regionX << 1)
    let srcRowNS = regionY * src.width + regionX
    let dstRowNS = regionY * dst.width + regionX

    for (let y = regionY; y < maxY; y++) {
      dstCells.set(srcCells.subarray(srcRowCI, srcRowCI + rowBytes), dstRowCI)
      dstNoSel.set(srcNoSel.subarray(srcRowNS, srcRowNS + rowLen), dstRowNS)
      dstWritten.set(srcWritten.subarray(srcRowNS, srcRowNS + rowLen), dstRowNS)
      srcRowCI += srcStride
      dstRowCI += dstStride
      srcRowNS += src.width
      dstRowNS += dst.width
    }
  }

  // Compute damage once for the whole region
  const regionRect = {
    x: regionX,
    y: regionY,
    width: rowLen,
    height: maxY - regionY
  }

  if (dst.damage) {
    dst.damage = unionRect(dst.damage, regionRect)
  } else {
    dst.damage = regionRect
  }

  // Handle wide char at right edge: spacer might be outside blit region
  // but still within dst bounds. Per-row check only at the boundary column.
  if (maxX < dst.width) {
    let srcLastCI = (regionY * src.width + (maxX - 1)) << 1
    let dstSpacerCI = (regionY * dst.width + maxX) << 1
    let wroteSpacerOutsideRegion = false

    for (let y = regionY; y < maxY; y++) {
      if ((srcCells[srcLastCI + 1]! & WIDTH_MASK) === CellWidth.Wide) {
        dstCells[dstSpacerCI] = SPACER_CHAR_INDEX
        dstCells[dstSpacerCI + 1] = packWord1(dst.emptyStyleId, 0, CellWidth.SpacerTail)
        dstWritten[y * dst.width + maxX] = 1
        wroteSpacerOutsideRegion = true
      }

      srcLastCI += srcStride
      dstSpacerCI += dstStride
    }

    // Expand damage to include SpacerTail column if we wrote any
    if (wroteSpacerOutsideRegion && dst.damage) {
      const rightEdge = dst.damage.x + dst.damage.width

      if (rightEdge === maxX) {
        dst.damage = { ...dst.damage, width: dst.damage.width + 1 }
      }
    }
  }
}

/**
 * Bulk-clear a rectangular region of the screen.
 * Uses BigInt64Array.fill() for fast row clears.
 * Handles wide character boundary cleanup at region edges.
 */
export function clearRegion(
  screen: Screen,
  regionX: number,
  regionY: number,
  regionWidth: number,
  regionHeight: number
): void {
  const startX = Math.max(0, regionX)
  const startY = Math.max(0, regionY)
  const maxX = Math.min(regionX + regionWidth, screen.width)
  const maxY = Math.min(regionY + regionHeight, screen.height)

  if (startX >= maxX || startY >= maxY) {
    return
  }

  const cells = screen.cells
  const cells64 = screen.cells64
  const written = screen.written
  const screenWidth = screen.width
  const rowBase = startY * screenWidth
  let damageMinX = startX
  let damageMaxX = maxX

  // EMPTY_CELL_VALUE (0n) matches the zero-initialized state:
  // word0=EMPTY_CHAR_INDEX(0), word1=packWord1(0,0,0)=0
  if (startX === 0 && maxX === screenWidth) {
    // Full-width: single fill, no boundary checks needed
    cells64.fill(EMPTY_CELL_VALUE, rowBase, rowBase + (maxY - startY) * screenWidth)
    written.fill(0, rowBase, rowBase + (maxY - startY) * screenWidth)
  } else {
    // Partial-width: single loop handles boundary cleanup and fill per row.
    const stride = screenWidth << 1 // 2 Int32s per cell
    const rowLen = maxX - startX
    const checkLeft = startX > 0
    const checkRight = maxX < screenWidth
    let leftEdge = (rowBase + startX) << 1
    let rightEdge = (rowBase + maxX - 1) << 1
    let fillStart = rowBase + startX

    for (let y = startY; y < maxY; y++) {
      // Left boundary: if cell at startX is a SpacerTail, the Wide char
      // at startX-1 (outside the region) will be orphaned. Clear it.
      if (checkLeft) {
        // leftEdge points to word0 of cell at startX; +1 is its word1
        if ((cells[leftEdge + 1]! & WIDTH_MASK) === CellWidth.SpacerTail) {
          // word1 of cell at startX-1 is leftEdge-1; word0 is leftEdge-2
          const prevW1 = leftEdge - 1

          if ((cells[prevW1]! & WIDTH_MASK) === CellWidth.Wide) {
            cells[prevW1 - 1] = EMPTY_CHAR_INDEX
            cells[prevW1] = packWord1(screen.emptyStyleId, 0, CellWidth.Narrow)
            written[y * screenWidth + startX - 1] = 0
            damageMinX = startX - 1
          }
        }
      }

      // Right boundary: if cell at maxX-1 is Wide, its SpacerTail at maxX
      // (outside the region) will be orphaned. Clear it.
      if (checkRight) {
        // rightEdge points to word0 of cell at maxX-1; +1 is its word1
        if ((cells[rightEdge + 1]! & WIDTH_MASK) === CellWidth.Wide) {
          // word1 of cell at maxX is rightEdge+3 (+2 to next word0, +1 to word1)
          const nextW1 = rightEdge + 3

          if ((cells[nextW1]! & WIDTH_MASK) === CellWidth.SpacerTail) {
            cells[nextW1 - 1] = EMPTY_CHAR_INDEX
            cells[nextW1] = packWord1(screen.emptyStyleId, 0, CellWidth.Narrow)
            written[y * screenWidth + maxX] = 0
            damageMaxX = maxX + 1
          }
        }
      }

      cells64.fill(EMPTY_CELL_VALUE, fillStart, fillStart + rowLen)
      written.fill(0, fillStart, fillStart + rowLen)
      leftEdge += stride
      rightEdge += stride
      fillStart += screenWidth
    }
  }

  // Update damage once for the whole region
  const regionRect = {
    x: damageMinX,
    y: startY,
    width: damageMaxX - damageMinX,
    height: maxY - startY
  }

  if (screen.damage) {
    screen.damage = unionRect(screen.damage, regionRect)
  } else {
    screen.damage = regionRect
  }
}

/**
 * Shift full-width rows within [top, bottom] (inclusive, 0-indexed) by n.
 * n > 0 shifts UP (simulating CSI n S); n < 0 shifts DOWN (CSI n T).
 * Vacated rows are cleared. Does NOT update damage. Both cells and the
 * noSelect bitmap are shifted so text-selection markers stay aligned when
 * this is applied to next.screen during scroll fast path.
 */
export function shiftRows(screen: Screen, top: number, bottom: number, n: number): void {
  if (n === 0 || top < 0 || bottom >= screen.height || top > bottom) {
    return
  }

  const w = screen.width
  const cells64 = screen.cells64
  const noSel = screen.noSelect
  const written = screen.written
  const sw = screen.softWrap
  const absN = Math.abs(n)

  if (absN > bottom - top) {
    cells64.fill(EMPTY_CELL_VALUE, top * w, (bottom + 1) * w)
    noSel.fill(0, top * w, (bottom + 1) * w)
    written.fill(0, top * w, (bottom + 1) * w)
    sw.fill(0, top, bottom + 1)

    return
  }

  if (n > 0) {
    // SU: row top+n..bottom → top..bottom-n; clear bottom-n+1..bottom
    cells64.copyWithin(top * w, (top + n) * w, (bottom + 1) * w)
    noSel.copyWithin(top * w, (top + n) * w, (bottom + 1) * w)
    written.copyWithin(top * w, (top + n) * w, (bottom + 1) * w)
    sw.copyWithin(top, top + n, bottom + 1)
    cells64.fill(EMPTY_CELL_VALUE, (bottom - n + 1) * w, (bottom + 1) * w)
    noSel.fill(0, (bottom - n + 1) * w, (bottom + 1) * w)
    written.fill(0, (bottom - n + 1) * w, (bottom + 1) * w)
    sw.fill(0, bottom - n + 1, bottom + 1)
  } else {
    // SD: row top..bottom+n → top-n..bottom; clear top..top-n-1
    cells64.copyWithin((top - n) * w, top * w, (bottom + n + 1) * w)
    noSel.copyWithin((top - n) * w, top * w, (bottom + n + 1) * w)
    written.copyWithin((top - n) * w, top * w, (bottom + n + 1) * w)
    sw.copyWithin(top - n, top, bottom + n + 1)
    cells64.fill(EMPTY_CELL_VALUE, top * w, (top - n) * w)
    noSel.fill(0, top * w, (top - n) * w)
    written.fill(0, top * w, (top - n) * w)
    sw.fill(0, top, top - n)
  }
}

// Matches OSC 8 ; ; URI BEL
const OSC8_REGEX = new RegExp(`^${ESC}\\]8${SEP}${SEP}([^${BEL}]*)${BEL}$`)
// OSC8 prefix: ESC ] 8 ; — cheap check to skip regex for the vast majority of styles (SGR = ESC [)
export const OSC8_PREFIX = `${ESC}]8${SEP}`

export function extractHyperlinkFromStyles(styles: AnsiCode[]): Hyperlink | null {
  for (const style of styles) {
    const code = style.code

    if (code.length < 5 || !code.startsWith(OSC8_PREFIX)) {
      continue
    }

    const match = code.match(OSC8_REGEX)

    if (match) {
      return match[1] || null
    }
  }

  return null
}

export function filterOutHyperlinkStyles(styles: AnsiCode[]): AnsiCode[] {
  return styles.filter(style => !style.code.startsWith(OSC8_PREFIX) || !OSC8_REGEX.test(style.code))
}

// ---

/**
 * Returns an array of all changes between two screens. Used by tests.
 * Production code should use diffEach() to avoid allocations.
 */
export function diff(prev: Screen, next: Screen): [point: Point, removed: Cell | undefined, added: Cell | undefined][] {
  const output: [Point, Cell | undefined, Cell | undefined][] = []
  diffEach(prev, next, (x, y, removed, added) => {
    // Copy cells since diffEach reuses the objects
    output.push([{ x, y }, removed ? { ...removed } : undefined, added ? { ...added } : undefined])
  })

  return output
}

type DiffCallback = (x: number, y: number, removed: Cell | undefined, added: Cell | undefined) => boolean | void

/**
 * Like diff(), but calls a callback for each change instead of building an array.
 * Reuses two Cell objects to avoid per-change allocations. The callback must not
 * retain references to the Cell objects — their contents are overwritten each call.
 *
 * Returns true if the callback ever returned true (early exit signal).
 */
export function diffEach(prev: Screen, next: Screen, cb: DiffCallback): boolean {
  const prevWidth = prev.width
  const nextWidth = next.width
  const prevHeight = prev.height
  const nextHeight = next.height

  let region: Rectangle

  if (prevWidth === 0 && prevHeight === 0) {
    region = { x: 0, y: 0, width: nextWidth, height: nextHeight }
  } else if (next.damage) {
    region = next.damage

    if (prev.damage) {
      region = unionRect(region, prev.damage)
    }
  } else if (prev.damage) {
    region = prev.damage
  } else {
    region = { x: 0, y: 0, width: 0, height: 0 }
  }

  if (prevHeight > nextHeight) {
    region = unionRect(region, {
      x: 0,
      y: nextHeight,
      width: prevWidth,
      height: prevHeight - nextHeight
    })
  }

  if (prevWidth > nextWidth) {
    region = unionRect(region, {
      x: nextWidth,
      y: 0,
      width: prevWidth - nextWidth,
      height: prevHeight
    })
  }

  const maxHeight = Math.max(prevHeight, nextHeight)
  const maxWidth = Math.max(prevWidth, nextWidth)
  const endY = Math.min(region.y + region.height, maxHeight)
  const endX = Math.min(region.x + region.width, maxWidth)

  if (prevWidth === nextWidth) {
    return diffSameWidth(prev, next, region.x, endX, region.y, endY, cb)
  }

  return diffDifferentWidth(prev, next, region.x, endX, region.y, endY, cb)
}

/**
 * Scan for the next cell that differs between two Int32Arrays.
 * Returns the number of matching cells before the first difference,
 * or `count` if all cells match. Tiny and pure for JIT inlining.
 */
function findNextDiff(a: Int32Array, b: Int32Array, w0: number, count: number): number {
  for (let i = 0; i < count; i++, w0 += 2) {
    const w1 = w0 | 1

    if (a[w0] !== b[w0] || a[w1] !== b[w1]) {
      return i
    }
  }

  return count
}

/**
 * Diff one row where both screens are in bounds.
 * Scans for differences with findNextDiff, unpacks and calls cb for each.
 */
function diffRowBoth(
  prevCells: Int32Array,
  nextCells: Int32Array,
  prev: Screen,
  next: Screen,
  ci: number,
  y: number,
  startX: number,
  endX: number,
  prevCell: Cell,
  nextCell: Cell,
  cb: DiffCallback
): boolean {
  let x = startX

  while (x < endX) {
    const skip = findNextDiff(prevCells, nextCells, ci, endX - x)
    x += skip
    ci += skip << 1

    if (x >= endX) {
      break
    }

    cellAtCI(prev, ci, prevCell)
    cellAtCI(next, ci, nextCell)

    if (cb(x, y, prevCell, nextCell)) {
      return true
    }

    x++
    ci += 2
  }

  return false
}

/**
 * Emit removals for a row that only exists in prev (height shrank).
 * Cannot skip empty cells — the terminal still has content from the
 * previous frame that needs to be cleared.
 */
function diffRowRemoved(
  prev: Screen,
  ci: number,
  y: number,
  startX: number,
  endX: number,
  prevCell: Cell,
  cb: DiffCallback
): boolean {
  for (let x = startX; x < endX; x++, ci += 2) {
    cellAtCI(prev, ci, prevCell)

    if (cb(x, y, prevCell, undefined)) {
      return true
    }
  }

  return false
}

/**
 * Emit additions for a row that only exists in next (height grew).
 * Skips empty/unwritten cells.
 */
function diffRowAdded(
  nextCells: Int32Array,
  next: Screen,
  ci: number,
  y: number,
  startX: number,
  endX: number,
  nextCell: Cell,
  cb: DiffCallback
): boolean {
  for (let x = startX; x < endX; x++, ci += 2) {
    if (nextCells[ci] === 0 && nextCells[ci | 1] === 0) {
      continue
    }

    cellAtCI(next, ci, nextCell)

    if (cb(x, y, undefined, nextCell)) {
      return true
    }
  }

  return false
}

/**
 * Diff two screens with identical width.
 * Dispatches each row to a small, JIT-friendly function.
 */
function diffSameWidth(
  prev: Screen,
  next: Screen,
  startX: number,
  endX: number,
  startY: number,
  endY: number,
  cb: DiffCallback
): boolean {
  const prevCells = prev.cells
  const nextCells = next.cells
  const width = prev.width
  const prevHeight = prev.height
  const nextHeight = next.height
  const stride = width << 1

  const prevCell: Cell = {
    char: ' ',
    styleId: 0,
    width: CellWidth.Narrow,
    hyperlink: undefined
  }

  const nextCell: Cell = {
    char: ' ',
    styleId: 0,
    width: CellWidth.Narrow,
    hyperlink: undefined
  }

  const rowEndX = Math.min(endX, width)
  let rowCI = (startY * width + startX) << 1

  for (let y = startY; y < endY; y++) {
    const prevIn = y < prevHeight
    const nextIn = y < nextHeight

    if (prevIn && nextIn) {
      if (diffRowBoth(prevCells, nextCells, prev, next, rowCI, y, startX, rowEndX, prevCell, nextCell, cb)) {
        return true
      }
    } else if (prevIn) {
      if (diffRowRemoved(prev, rowCI, y, startX, rowEndX, prevCell, cb)) {
        return true
      }
    } else if (nextIn) {
      if (diffRowAdded(nextCells, next, rowCI, y, startX, rowEndX, nextCell, cb)) {
        return true
      }
    }

    rowCI += stride
  }

  return false
}

/**
 * Fallback: diff two screens with different widths (resize).
 * Separate indices for prev and next cells arrays.
 */
function diffDifferentWidth(
  prev: Screen,
  next: Screen,
  startX: number,
  endX: number,
  startY: number,
  endY: number,
  cb: DiffCallback
): boolean {
  const prevWidth = prev.width
  const nextWidth = next.width
  const prevCells = prev.cells
  const nextCells = next.cells

  const prevCell: Cell = {
    char: ' ',
    styleId: 0,
    width: CellWidth.Narrow,
    hyperlink: undefined
  }

  const nextCell: Cell = {
    char: ' ',
    styleId: 0,
    width: CellWidth.Narrow,
    hyperlink: undefined
  }

  const prevStride = prevWidth << 1
  const nextStride = nextWidth << 1
  let prevRowCI = (startY * prevWidth + startX) << 1
  let nextRowCI = (startY * nextWidth + startX) << 1

  for (let y = startY; y < endY; y++) {
    const prevIn = y < prev.height
    const nextIn = y < next.height
    const prevEndX = prevIn ? Math.min(endX, prevWidth) : startX
    const nextEndX = nextIn ? Math.min(endX, nextWidth) : startX
    const bothEndX = Math.min(prevEndX, nextEndX)

    let prevCI = prevRowCI
    let nextCI = nextRowCI

    for (let x = startX; x < bothEndX; x++) {
      if (prevCells[prevCI] === nextCells[nextCI] && prevCells[prevCI + 1] === nextCells[nextCI + 1]) {
        prevCI += 2
        nextCI += 2

        continue
      }

      cellAtCI(prev, prevCI, prevCell)
      cellAtCI(next, nextCI, nextCell)
      prevCI += 2
      nextCI += 2

      if (cb(x, y, prevCell, nextCell)) {
        return true
      }
    }

    if (prevEndX > bothEndX) {
      prevCI = prevRowCI + ((bothEndX - startX) << 1)

      for (let x = bothEndX; x < prevEndX; x++) {
        cellAtCI(prev, prevCI, prevCell)
        prevCI += 2

        if (cb(x, y, prevCell, undefined)) {
          return true
        }
      }
    }

    if (nextEndX > bothEndX) {
      nextCI = nextRowCI + ((bothEndX - startX) << 1)

      for (let x = bothEndX; x < nextEndX; x++) {
        if (nextCells[nextCI] === 0 && nextCells[nextCI | 1] === 0) {
          nextCI += 2

          continue
        }

        cellAtCI(next, nextCI, nextCell)
        nextCI += 2

        if (cb(x, y, undefined, nextCell)) {
          return true
        }
      }
    }

    prevRowCI += prevStride
    nextRowCI += nextStride
  }

  return false
}

/**
 * Mark a rectangular region as noSelect (exclude from text selection).
 * Clamps to screen bounds. Called from output.ts when a <NoSelect> box
 * renders. No damage tracking — noSelect doesn't affect terminal output,
 * only getSelectedText/applySelectionOverlay which read it directly.
 */
export function markNoSelectRegion(screen: Screen, x: number, y: number, width: number, height: number): void {
  const maxX = Math.min(x + width, screen.width)
  const maxY = Math.min(y + height, screen.height)
  const noSel = screen.noSelect
  const stride = screen.width

  for (let row = Math.max(0, y); row < maxY; row++) {
    const rowStart = row * stride
    noSel.fill(1, rowStart + Math.max(0, x), rowStart + maxX)
  }
}
