/** xterm SGR mouse: `\x1b[<button;col;rowM|m`. Button 0–2 is click;
 * 32+ is motion, 64+ is wheel. Clicks must reach Ink; the rest must not. */
export function shouldDropPtyMouseReport(data: string): boolean {
  const match = /^\x1b\[<(\d+);\d+;\d+[Mm]$/.exec(data);
  if (!match) return false;
  return Number(match[1]) >= 32;
}
