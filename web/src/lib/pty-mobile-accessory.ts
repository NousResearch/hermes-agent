/** Terminus-style keys above the iPhone software keyboard. */

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

/** `position:fixed` is the visual viewport on iOS — sit on the keyboard, not under it. */
export function accessoryDockBottomPx(_keyboardInsetPx: number): number {
  return 0;
}

export function mountPtyMobileAccessory(
  _host: HTMLElement,
  actions: { paste: () => void; interrupt: () => void; caretLeft: () => void; caretRight: () => void },
): { setInset: (insetPx: number, coarsePointer: boolean, focused?: boolean) => void; dispose: () => void } {
  const doc = _host.ownerDocument;
  const bar = doc.createElement("div");
  bar.className = "pty-mobile-accessory";
  bar.setAttribute("role", "toolbar");
  bar.setAttribute("aria-label", "Terminal keys");
  bar.style.display = "none";
  bar.style.position = "fixed";
  bar.style.flexWrap = "wrap";
  bar.style.alignItems = "center";
  bar.style.left = "0";
  bar.style.right = "0";
  bar.style.bottom = "0";
  bar.style.zIndex = "2147483646";
  bar.style.gap = "8px";
  bar.style.padding = "8px 10px";
  bar.style.paddingBottom = "max(8px, env(safe-area-inset-bottom))";
  bar.style.background = "rgba(20,20,20,0.96)";
  bar.style.borderTop = "1px solid rgba(255,255,255,0.18)";
  bar.style.boxSizing = "border-box";
  const mk = (label: string, onClick: () => void) => {
    const btn = doc.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.style.minHeight = "40px";
    btn.style.padding = "0 14px";
    btn.style.borderRadius = "8px";
    btn.style.border = "1px solid rgba(255,255,255,0.25)";
    btn.style.background = "#2a2a2a";
    btn.style.color = "#f5f5f5";
    btn.style.font = "600 15px/1 system-ui,sans-serif";
    btn.addEventListener("pointerdown", (event) => {
      event.preventDefault();
    });
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
  mk("Left", actions.caretLeft);
  mk("Right", actions.caretRight);
  mk("Paste", actions.paste);
  mk("Ctrl+C", actions.interrupt);
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
