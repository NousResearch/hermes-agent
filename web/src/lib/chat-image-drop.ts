import { imageFilesFromTransfer, transferMayContainImage } from "./chatImagePaste";

/**
 * Browser clipboard/drag-and-drop image wiring for the chat host element.
 *
 * paste / dragover / drop listeners (capture phase) that pull image files out
 * of the transfer data and hand them to the caller's upload+attach pipeline.
 * `attachChatImageDropListeners` returns a detach function for effect cleanup.
 */
export function attachChatImageDropListeners(
  host: HTMLElement,
  uploadAndAttachImages: (files: File[]) => void,
): () => void {
  const handleBrowserPaste = (ev: ClipboardEvent) => {
    const files = imageFilesFromTransfer(ev.clipboardData);
    if (!files.length) return;
    ev.preventDefault();
    ev.stopPropagation();
    uploadAndAttachImages(files);
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
