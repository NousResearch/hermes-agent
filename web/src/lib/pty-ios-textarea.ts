/** Make xterm's helper textarea eligible for Safari dictation, paste, and
 * the iPhone space-bar trackpad. A 24px strip compresses the whole line into
 * a few millimetres, so the caret cannot be placed in the middle. */
export function preparePtyTextareaForDictation(textarea: HTMLTextAreaElement): void {
  textarea.setAttribute("autocomplete", "on");
  textarea.setAttribute("autocorrect", "on");
  textarea.setAttribute("autocapitalize", "sentences");
  textarea.setAttribute("spellcheck", "true");
  textarea.setAttribute("enterkeyhint", "send");
  textarea.setAttribute("inputmode", "text");
  textarea.removeAttribute("readonly");
  textarea.readOnly = false;
  textarea.disabled = false;
  textarea.style.opacity = "0.01";
  textarea.style.fontSize = "16px";
  textarea.style.width = "100%";
  textarea.style.height = "100%";
  textarea.style.minWidth = "1px";
  textarea.style.minHeight = "24px";
  textarea.style.position = "absolute";
  textarea.style.left = "0";
  textarea.style.right = "0";
  textarea.style.top = "0";
  textarea.style.bottom = "0";
  textarea.style.zIndex = "2";
  textarea.style.overflow = "hidden";
}
