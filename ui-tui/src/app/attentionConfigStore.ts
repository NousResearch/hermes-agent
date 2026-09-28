// display.notify_on_interact / display.tui_attention_hook — module-level
// flags rather than hook-state threading, because event and server-request
// handlers read them outside React (same shape as wakeState.ts). useConfigSync
// refreshes them from `config.get full`, so live edits apply without a
// restart; before the first fetch both stay at their disabled defaults.
export interface TuiAttentionHookConfig {
  command: string
  enabled: boolean
}

let notifyOnInteract = false
let attentionHook: TuiAttentionHookConfig = { command: '', enabled: false }

export const getNotifyOnInteract = (): boolean => notifyOnInteract

export const setNotifyOnInteract = (v: boolean): void => {
  notifyOnInteract = v
}

export const getAttentionHook = (): TuiAttentionHookConfig => attentionHook

export const setAttentionHook = (cfg: TuiAttentionHookConfig): void => {
  attentionHook = cfg
}

/** Test-only: restore both flags to their disabled defaults. */
export const resetAttentionConfigForTests = (): void => {
  notifyOnInteract = false
  attentionHook = { command: '', enabled: false }
}
