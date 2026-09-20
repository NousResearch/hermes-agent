/** Safari dictation needs xterm's helper textarea to stay in the layout.
 *
 * Hiding it the usual ways — `opacity: 0`, `text-indent: -9999px`, parking it
 * off-screen — makes Safari treat it as an unreachable field and garble
 * dictated text. Keeping it in-layout means Safari also paints autocorrect
 * candidates and replacement glyphs there, in black, on top of Ink. So the
 * helper covers the composer row at full width and is made fully transparent
 * instead of hidden.
 */

export interface PtyTextareaBox {
  top: number;
  height: number;
}

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

/** The bottom row of the screen — where Ink draws its input line. */
export function composerTextareaBox(rows: number, screenHeight: number): PtyTextareaBox {
  const safeRows = Math.max(1, rows);
  const height = screenHeight > 0 ? screenHeight / safeRows : 24;
  return { top: height * (safeRows - 1), height };
}

export function restorePtyTextareaLayout(
  textarea: HTMLTextAreaElement,
  box?: PtyTextareaBox,
): void {
  const height = box?.height ?? 24;
  const top = box?.top ?? 0;
  textarea.style.opacity = "0.01";
  textarea.style.setProperty("color", "transparent", "important");
  textarea.style.setProperty("caret-color", "transparent", "important");
  textarea.style.setProperty("-webkit-text-fill-color", "transparent", "important");
  textarea.style.background = "transparent";
  // 16px is the threshold below which iOS zooms the page on focus.
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
  // xterm focuses the helper on every click; without preventScroll Safari
  // scrolls the page to it and the transcript jumps.
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
  // Declared after `apply` so it can stay const; `apply` only runs once the
  // observer exists.
  const observer = new MutationObserver(apply);
  observer.observe(textarea, { attributes: true, attributeFilter: ["style"] });
  ensurePtyHelperInkCss(textarea.ownerDocument);
  textarea.addEventListener("input", apply);
  apply();
  return () => {
    observer.disconnect();
    textarea.removeEventListener("input", apply);
  };
}
