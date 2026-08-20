export interface PtyWheelModifiers {
  altKey?: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
}

/** Encode a browser wheel event as an xterm SGR mouse report.
 *
 * The embedded TUI consumes button 64/65 as wheelUp/wheelDown and applies its
 * own row-level acceleration. Coordinates are intentionally pinned to 1,1:
 * transcript scrolling is handled globally and does not depend on a hit target.
 */
export function encodePtyWheel(
  deltaY: number,
  modifiers: PtyWheelModifiers = {},
): string | null {
  if (!Number.isFinite(deltaY) || deltaY === 0) return null;

  const modifierBits =
    (modifiers.shiftKey ? 4 : 0) +
    (modifiers.altKey || modifiers.metaKey ? 8 : 0) +
    (modifiers.ctrlKey ? 16 : 0);
  const button = (deltaY < 0 ? 64 : 65) + modifierBits;

  return `\u001b[<${button};1;1M`;
}
