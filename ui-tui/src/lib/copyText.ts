import { setClipboard } from '@hermes/ink'

export type CopyTextOutcome =
  | { method: 'native-or-tmux'; success: true }
  | { method: 'osc52'; success: true }
  | { method: 'none'; success: false }

/**
 * Shared application-level clipboard wrapper.
 *
 * Calls `setClipboard`, emits the returned terminal sequence, and returns a
 * typed outcome. Never transforms or logs `text`.
 */
export async function copyText(text: string): Promise<CopyTextOutcome> {
  try {
    const result = await setClipboard(text)

    if (result.sequence.length > 0) {
      process.stdout.write(result.sequence)
    }

    if (result.success) {
      // native path (pbcopy/wl-copy/etc.) succeeded, or tmux buffer loaded
      const method = result.sequence.length > 0 ? 'osc52' : 'native-or-tmux'

      return { method, success: true }
    }

    return { method: 'none', success: false }
  } catch {
    return { method: 'none', success: false }
  }
}
