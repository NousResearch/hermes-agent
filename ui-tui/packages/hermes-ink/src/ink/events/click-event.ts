import { Event } from './event.js'

/**
 * Mouse click event. Fired on left-button release without drag, only when
 * mouse tracking is enabled (i.e. inside <AlternateScreen>).
 *
 * Bubbles from the deepest hit node up through parentNode. Call
 * stopImmediatePropagation() to prevent ancestors' onClick from firing.
 */
export class ClickEvent extends Event {
  /** 0-indexed screen column of the click */
  readonly col: number
  /** 0-indexed screen row of the click */
  readonly row: number
  /**
   * Click column relative to the current handler's Box (col - box.x).
   * Recomputed by dispatchClick before each handler fires, so an onClick
   * on a container sees coords relative to that container, not to any
   * child the click landed on.
   */
  localCol = 0
  /** Click row relative to the current handler's Box (row - box.y). */
  localRow = 0
  /**
   * True if the clicked cell has no visible content (unwritten in the
   * screen buffer — both packed words are 0). Handlers can check this to
   * ignore clicks on blank space to the right of text, so accidental
   * clicks on empty terminal space don't toggle state.
   */
  readonly cellIsBlank: boolean

  constructor(col: number, row: number, cellIsBlank: boolean) {
    super()
    this.col = col
    this.row = row
    this.cellIsBlank = cellIsBlank
  }
}
