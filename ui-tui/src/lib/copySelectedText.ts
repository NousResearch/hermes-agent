import type { CopyDelivery } from '../app/copyFeedbackStore.js'

import { writeClipboardText } from './clipboard.js'
import { writeOsc52Clipboard } from './osc52.js'
import { isRemoteShell } from './platform.js'

/** Composer selections need the same remote-terminal delivery as transcript selections. */
export async function copySelectedText(text: string): Promise<CopyDelivery> {
  if (isRemoteShell() || !(await writeClipboardText(text))) {
    writeOsc52Clipboard(text)

    return 'terminal'
  }

  return 'clipboard'
}
