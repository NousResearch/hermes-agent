import { Event } from './event.js'

export class MouseEvent extends Event {
  readonly col: number
  readonly row: number
  localCol = 0
  localRow = 0
  readonly cellIsBlank: boolean
  readonly button: number

  constructor(col: number, row: number, cellIsBlank: boolean, button: number) {
    super()
    this.col = col
    this.row = row
    this.cellIsBlank = cellIsBlank
    this.button = button
  }
}
