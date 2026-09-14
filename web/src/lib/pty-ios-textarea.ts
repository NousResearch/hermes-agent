/** Safari dictation/trackpad need the helper in-layout.
 * Desktop: cover the composer row (transparent).
 * Phone: dock a real visible field above the accessory bar so caret and
 * transcript scroll are separate surfaces. */

export const MOBILE_COMPOSER_HEIGHT_PX = 44;

export type PtyTextareaBox =
  | { top: number; height: number; dock?: undefined }
  | { dock: number; top?: undefined; height?: undefined };

export const PTY_HELPER_INK_STYLE_ID = "pty-helper-ink";

/** Survives xterm rewriting the helper's inline style (Safari autocorrect paints black otherwise). */
export function ensurePtyHelperInkCss(doc: Document): void {
  const css = [
    ".xterm textarea.xterm-helper-textarea,.xterm .xterm-helper-textarea{",
    "color:transparent!important;",
    "caret-color:transparent!important;",
    "-webkit-text-fill-color:transparent!important;",
    "background:transparent!important;",
    "opacity:0.01!important;",
    "text-shadow:none!important;",
    "z-index:2!important;",
    "}",
    ".xterm .composition-view,.xterm .composition-view.active{",
    "color:transparent!important;",
    "-webkit-text-fill-color:transparent!important;",
    "background:transparent!important;",
    "text-shadow:none!important;",
    "}",
  ].join("");
  const existing = doc.getElementById(PTY_HELPER_INK_STYLE_ID);
  if (existing) {
    existing.textContent = css;
    return;
  }
  const style = doc.createElement("style");
  style.id = PTY_HELPER_INK_STYLE_ID;
  style.textContent = css;
  (doc.head ?? doc.documentElement).append(style);
}

export function composerTextareaBox(rows: number, screenHeight: number): PtyTextareaBox {
  const safeRows = Math.max(1, rows);
  const height = screenHeight > 0 ? screenHeight / safeRows : 24;
  return { top: height * (safeRows - 1), height };
}

export function dockMobileComposer(textarea: HTMLTextAreaElement, bottomPx: number): void {
  const bottom = Number.isFinite(bottomPx) && bottomPx > 0 ? Math.round(bottomPx) : 0;
  textarea.style.cssText = [
    "position:fixed !important",
    "left:0 !important",
    "right:0 !important",
    "top:auto !important",
    `bottom:${bottom}px !important`,
    "width:100% !important",
    `height:${MOBILE_COMPOSER_HEIGHT_PX}px !important`,
    `max-height:${MOBILE_COMPOSER_HEIGHT_PX}px !important`,
    `min-height:${MOBILE_COMPOSER_HEIGHT_PX}px !important`,
    "opacity:1 !important",
    "color:#f5f5f5 !important",
    "caret-color:#fff !important",
    "-webkit-text-fill-color:#f5f5f5 !important",
    "background:#1a1a1a !important",
    "font-size:16px !important",
    "line-height:22px !important",
    "padding:10px 12px !important",
    "z-index:2147483645 !important",
    "border:0 !important",
    "border-top:1px solid rgba(255,255,255,0.18) !important",
    "overflow:auto !important",
    "white-space:pre-wrap !important",
    "pointer-events:auto !important",
    "box-sizing:border-box !important",
    "margin:0 !important",
  ].join(";");
}

export function restorePtyTextareaLayout(
  textarea: HTMLTextAreaElement,
  box?: PtyTextareaBox,
): void {
  if (box && typeof box.dock === "number") {
    dockMobileComposer(textarea, box.dock);
    return;
  }
  const height = box?.height ?? 24;
  const top = box?.top ?? 0;
  textarea.style.opacity = "0.01";
  textarea.style.setProperty("color", "transparent", "important");
  textarea.style.setProperty("caret-color", "transparent", "important");
  textarea.style.setProperty("-webkit-text-fill-color", "transparent", "important");
  textarea.style.background = "transparent";
  textarea.style.fontSize = `${Math.max(16, height)}px`;
  textarea.style.lineHeight = `${height}px`;
  textarea.style.whiteSpace = "pre";
  textarea.style.width = "100%";
  textarea.style.height = `${height}px`;
  textarea.style.minWidth = "1px";
  textarea.style.minHeight = `${height}px`;
  textarea.style.position = "absolute";
  textarea.style.left = "0";
  textarea.style.right = "0";
  textarea.style.top = `${top}px`;
  textarea.style.bottom = "auto";
  textarea.style.zIndex = "2";
  textarea.style.overflow = "hidden";
  textarea.style.removeProperty("text-indent");
  textarea.style.pointerEvents = "auto";
  textarea.style.padding = "0";
  textarea.style.border = "0";
}

export function preparePtyTextareaForDictation(textarea: HTMLTextAreaElement): void {
  ensurePtyHelperInkCss(textarea.ownerDocument);
  textarea.setAttribute("autocomplete", "off");
  textarea.setAttribute("autocorrect", "off");
  textarea.setAttribute("autocapitalize", "off");
  textarea.setAttribute("spellcheck", "false");
  textarea.setAttribute("enterkeyhint", "send");
  textarea.setAttribute("inputmode", "text");
  textarea.removeAttribute("readonly");
  textarea.readOnly = false;
  textarea.disabled = false;
  restorePtyTextareaLayout(textarea);
  const nativeFocus = textarea.focus.bind(textarea);
  textarea.focus = ((options?: FocusOptions) => {
    nativeFocus({ ...options, preventScroll: true });
  }) as typeof textarea.focus;
}

/** xterm writes left/top/width/height onto the helper each frame. */
export function watchPtyTextareaLayout(
  textarea: HTMLTextAreaElement,
  box?: () => PtyTextareaBox,
): () => void {
  let applying = false;
  let observer: MutationObserver;
  const apply = () => {
    if (applying) return;
    applying = true;
    observer.disconnect();
    try {
      restorePtyTextareaLayout(textarea, box?.());
    } finally {
      observer.observe(textarea, { attributes: true, attributeFilter: ["style"] });
      applying = false;
    }
  };
  observer = new MutationObserver(apply);
  observer.observe(textarea, { attributes: true, attributeFilter: ["style"] });
  ensurePtyHelperInkCss(textarea.ownerDocument);
  textarea.addEventListener("input", apply);
  apply();
  return () => {
    observer.disconnect();
    textarea.removeEventListener("input", apply);
  };
}
