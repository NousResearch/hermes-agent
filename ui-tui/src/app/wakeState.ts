// Session-scoped memory of an explicit `/wake off`.
//
// The gateway auto-arms the "Hey Hermes" listener on every `gateway.ready`
// (see createGatewayEventHandler.ts). When the user explicitly disables the
// listener with `/wake off`, a reconnect must NOT silently re-arm it — this
// module-level flag records that intent for the lifetime of the process.
// `/wake on` clears it. Deliberately not persisted: config (`wake_word.*`)
// remains the durable on/off switch; this is only per-session steering.
let wakeUserDisabled = false

export const isWakeUserDisabled = (): boolean => wakeUserDisabled

export const setWakeUserDisabled = (disabled: boolean): void => {
  wakeUserDisabled = disabled
}
