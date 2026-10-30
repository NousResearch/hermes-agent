import { reportClipboardCopy, reportClipboardCut } from '../app/clipboardFeedbackStore.js'

/** Preserve the draft on failure; announce a cut only when the copied text was removed. */
export async function cutSelection(
  text: string,
  write: (text: string) => Promise<boolean>,
  removeSelection: () => boolean | void
): Promise<boolean> {
  const ok = await write(text)

  if (ok) {
    if (removeSelection() === false) {
      reportClipboardCopy(text)
    } else {
      reportClipboardCut(text)
    }
  }

  return ok
}
