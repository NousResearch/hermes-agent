import { setClipboard } from './termio/osc.js'

/** A receipt is emitted only after a clipboard path succeeded or OSC 52 was sent. */
export async function copySelectionText(
  text: string,
  writeSequence: (sequence: string) => unknown,
  listeners: Iterable<(text: string) => void>,
  copy = setClipboard
): Promise<string> {
  if (!text) {
    return ''
  }

  try {
    const { sequence, success } = await copy(text)

    if (sequence) {
      writeSequence(sequence)
    }

    if (success) {
      for (const notify of listeners) {
        notify(text)
      }

      return text
    }
  } catch {
    // The caller keeps the highlight and can retry another clipboard path.
  }

  return ''
}
