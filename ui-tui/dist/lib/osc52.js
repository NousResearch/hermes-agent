export const writeOsc52Clipboard = (s) => process.stdout.write(`\x1b]52;c;${Buffer.from(s, 'utf8').toString('base64')}\x07`);
