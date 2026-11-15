import { imageFilesFromTransfer, transferMayContainImage } from "./chatImagePaste";

/**
 * Browser clipboard/drag-and-drop image wiring for the chat host element.
 *
 * paste / dragover / drop listeners (capture phase) that pull image files out
 * of the transfer data and hand them to the caller's upload+attach pipeline.
 * `attachChatImageDropListeners` returns a detach function for effect cleanup.
 *
 * Plain-text pastes go through `pasteText` (#52471): xterm's own paste listener
 * clears the hidden textarea, but the browser's default insert runs AFTER event
 * dispatch, so the stale value survives and the next typed character gets
 * duplicated ("when" → "whenn"). Cancelling the native paste and delivering the
 * text through `term.paste()` exactly once is the same route the Ctrl/Cmd+V
 * keydown interception uses.
 */
export function attachChatImageDropListeners(
  host: HTMLElement,
  uploadAndAttachImages: (files: File[]) => void,
  pasteText: (text: string) => void,
): () => void {
  const handleBrowserPaste = (ev: ClipboardEvent) => {
    const files = imageFilesFromTransfer(ev.clipboardData);
    if (files.length) {
      ev.preventDefault();
      ev.stopPropagation();
      uploadAndAttachImages(files);
      return;
    }
    const text = ev.clipboardData?.getData("text/plain");
    if (text) {
      ev.preventDefault();
      ev.stopPropagation();
      pasteText(text);
    }
  };
  const handleBrowserDragOver = (ev: DragEvent) => {
    if (!transferMayContainImage(ev.dataTransfer)) return;
    ev.preventDefault();
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = "copy";
  };
  const handleBrowserDrop = (ev: DragEvent) => {
    const files = imageFilesFromTransfer(ev.dataTransfer);
    if (!files.length) return;
    ev.preventDefault();
    ev.stopPropagation();
    uploadAndAttachImages(files);
  };
  host.addEventListener("paste", handleBrowserPaste, { capture: true });
  host.addEventListener("dragover", handleBrowserDragOver, { capture: true });
  host.addEventListener("drop", handleBrowserDrop, { capture: true });
  return () => {
    host.removeEventListener("paste", handleBrowserPaste, true);
    host.removeEventListener("dragover", handleBrowserDragOver, true);
    host.removeEventListener("drop", handleBrowserDrop, true);
  };
}
