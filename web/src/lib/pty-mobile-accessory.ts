/** Terminus-style keys above the iPhone software keyboard. */

export const PTY_ARROW_UP = "\x1b[A";
export const PTY_ARROW_DOWN = "\x1b[B";
export const PTY_ETX = "\x03";
export const ACCESSORY_BAR_HEIGHT_PX = 56;

export function shouldShowMobileAccessory(
  keyboardInsetPx: number,
  coarsePointer: boolean,
  textareaFocused = false,
): boolean {
  if (!coarsePointer) return false;
  return keyboardInsetPx > 0 || textareaFocused;
}

/** Padding under xterm: keyboard overlay plus the accessory bar. */
export function terminalBottomReservePx(
  keyboardInsetPx: number,
  accessoryVisible: boolean,
): number {
  const inset = Number.isFinite(keyboardInsetPx) && keyboardInsetPx > 0 ? keyboardInsetPx : 0;
  return inset + (accessoryVisible ? ACCESSORY_BAR_HEIGHT_PX : 0);
}

/** Layout-viewport `position:fixed` sits under the iOS keyboard — offset by the inset. */
export function accessoryDockBottomPx(keyboardInsetPx: number): number {
  return Number.isFinite(keyboardInsetPx) && keyboardInsetPx > 0 ? Math.round(keyboardInsetPx) : 0;
}

/**
 * Dispatch a real xterm arrow key without moving focus away from the helper
 * textarea. The browser-input adapter normally owns Up/Down while its native
 * caret mirror is active. A synthetic contextmenu event switches that adapter
 * to helper-owned mode for this one key, allowing xterm to encode the arrow
 * and forward it through the normal PTY onData path.
 */
export function dispatchPtyArrowKey(
  host: HTMLElement,
  key: "ArrowUp" | "ArrowDown",
): boolean {
  const terminal = host.classList.contains("xterm")
    ? host
    : host.querySelector<HTMLElement>(".xterm");
  const textarea = terminal?.querySelector<HTMLTextAreaElement>(".xterm-helper-textarea");
  if (!terminal || !textarea) return false;

  terminal.dispatchEvent(new Event("contextmenu", { bubbles: true, cancelable: true }));
  if (textarea.ownerDocument.activeElement !== textarea) textarea.focus();

  const init: KeyboardEventInit = {
    key,
    code: key,
    bubbles: true,
    cancelable: true,
  };
  textarea.dispatchEvent(new KeyboardEvent("keydown", init));
  textarea.dispatchEvent(new KeyboardEvent("keyup", init));
  return true;
}

export function mountPtyMobileAccessory(
  _host: HTMLElement,
  actions: { paste: () => void; interrupt: () => void; caretLeft: () => void; caretRight: () => void; historyUp: () => void; historyDown: () => void },
): { setInset: (insetPx: number, coarsePointer: boolean, focused?: boolean) => void; dispose: () => void } {
  const doc = _host.ownerDocument;
  const bar = doc.createElement("div");
  bar.className = "pty-mobile-accessory";
  bar.setAttribute("role", "toolbar");
  bar.setAttribute("aria-label", "Terminal keys");
  bar.style.display = "none";
  bar.style.position = "fixed";
  bar.style.flexWrap = "nowrap";
  bar.style.alignItems = "center";
  bar.style.left = "0";
  bar.style.right = "0";
  bar.style.bottom = "0";
  bar.style.zIndex = "2147483646";
  bar.style.gap = "6px";
  bar.style.padding = "8px";
  bar.style.paddingBottom = "max(8px, env(safe-area-inset-bottom))";
  bar.style.background = "rgba(20,20,20,0.96)";
  bar.style.borderTop = "1px solid rgba(255,255,255,0.18)";
  bar.style.boxSizing = "border-box";
  bar.style.overflowX = "auto";
  bar.style.setProperty("-webkit-overflow-scrolling", "touch");

  const mk = (label: string, ariaLabel: string, onClick: () => void, compact = false) => {
    const btn = doc.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.setAttribute("aria-label", ariaLabel);
    btn.style.minHeight = "40px";
    btn.style.minWidth = compact ? "42px" : "auto";
    btn.style.padding = compact ? "0 10px" : "0 12px";
    btn.style.borderRadius = "8px";
    btn.style.border = "1px solid rgba(255,255,255,0.25)";
    btn.style.background = "#2a2a2a";
    btn.style.color = "#f5f5f5";
    btn.style.font = compact ? "700 19px/1 system-ui,sans-serif" : "600 14px/1 system-ui,sans-serif";
    btn.style.flex = "0 0 auto";
    btn.addEventListener("pointerdown", (event) => {
      event.preventDefault();
    });
    btn.addEventListener("touchstart", (event) => {
      event.preventDefault();
    }, { passive: false });
    let lastFire = 0;
    const fire = (event: Event) => {
      event.preventDefault();
      event.stopPropagation();
      const now = Date.now();
      if (now - lastFire < 80) return;
      lastFire = now;
      onClick();
    };
    btn.addEventListener("pointerup", fire);
    btn.addEventListener("click", fire);
    bar.append(btn);
    return btn;
  };

  mk("↑", "Previous command", actions.historyUp, true);
  mk("↓", "Next command", actions.historyDown, true);
  mk("←", "Arrow left", actions.caretLeft, true);
  mk("→", "Arrow right", actions.caretRight, true);
  mk("Paste", "Paste clipboard", actions.paste);
  mk("Ctrl+C", "Interrupt with Control C", actions.interrupt);
  doc.body.append(bar);

  const setInset = (insetPx: number, coarsePointer: boolean, focused = false) => {
    const show = shouldShowMobileAccessory(insetPx, coarsePointer, focused);
    bar.style.display = show ? "flex" : "none";
    bar.style.bottom = `${accessoryDockBottomPx(insetPx)}px`;
  };

  return {
    setInset,
    dispose() {
      bar.remove();
    },
  };
}
