// Encode Sixel into an escape sequence.
// For v2 we delegate Sixel encoding of arbitrary PNGs to the server side
// (the gateway pre-encodes the Sixel payload and ships it as a string).
// This file is a thin client wrapper that renders the pre-encoded string.

const DCS = '\x1bP'
const ST = '\x1b\\'

export interface SixelOptions {
  // Pre-encoded Sixel payload (string) from the server.
  payload: string
}

export function encodeSixel(opts: SixelOptions): string {
  // Sixel data starts with introducer + raster attributes,
  // then the payload, then String Terminator.
  return `${DCS}0;0;0;0q${opts.payload}${ST}`
}
