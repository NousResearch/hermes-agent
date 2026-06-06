// Encode a PNG into an iTerm2 inline image escape sequence.
// See: https://iterm2.com/documentation-images.html
// Format: OSC 1337 ; File=[name=base64name;size=N;width=N;height=N;inline=1]:base64 ST

const OSC = '\x1b]'
const ST = '\x1b\\'

export interface ITermOptions {
  width: number
  height: number
  name?: string
}

export function encodeITerm(png: Buffer, opts: ITermOptions): string {
  const b64 = png.toString('base64')
  const name = btoa(opts.name ?? 'image.png')
  const params = [
    `name=${name}`,
    `size=${png.length}`,
    `width=${opts.width}`,
    `height=${opts.height}`,
    'inline=1',
  ].join(';')
  return `${OSC}1337;File=${params}:${b64}${ST}`
}
