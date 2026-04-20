export const writeOsc52Clipboard = (s: string) =>
  process.stdout.write(`\x1b]52;c;${Buffer.from(s, 'utf8').toString('base64')}\x07`)
